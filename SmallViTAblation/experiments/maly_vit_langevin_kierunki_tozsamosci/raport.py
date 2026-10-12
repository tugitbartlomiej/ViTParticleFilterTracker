"""Raport zachowuje negatywy długich horyzontów i odróżnia znaki składowych od wektorów."""
import json,csv,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dane import HERE,read,js,sha,image
from prognoza import load_models

def interval(v,scale=100):return f'[{float(v[0])*scale:.2f}, {float(v[1])*scale:.2f}]'

def main():
    a=dict(np.load(HERE/'TRAJEKTORIE_PRZYCZYNOWE_K500.npz'));retro=dict(np.load(HERE/'TRAJEKTORIE_RETROSPEKTYWNE_K1000.npz'));data=read(HERE/'DANE.json');identity=read(HERE/'IDENTYFIKACJA_K500.json');alignment=read(HERE/'ZGODNOSC_K500.json')
    metrics={dt:read(HERE/f'METRYKI_DT{dt}.json') for dt in (1,10)};noise={dt:read(HERE/f'SZUM_DT{dt}.json') for dt in (1,10)};params={dt:load_models(dt) for dt in (1,10)}
    n=int(a['tool_count']);ids=a['token_ids'][:n];coverage=a['coverage600'][ids];q=a['q500'][:,:n];w=a['tool_weights'];origins=np.arange(800,951)
    boundary=int(np.argmin(coverage));interior=int(np.argmax(coverage));middle=int(np.argsort(coverage)[n//2]);chosen=list(dict.fromkeys([boundary,interior,middle]));colors=['#1c7fa0','#a82732','#278047']
    filters={}
    for dt in (1,10):
        mod=params[dt]['selected'];ori=np.array(metrics[dt]['origin_updates']);filters[str(dt)]={}
        for H in metrics[dt]['horizons']:
            actual=q[ori+H]-q[ori];valid=np.abs(actual)>mod['epsilon'];weighted=float(np.einsum('ond,n->',valid,w)/len(ori)/8)
            filters[str(dt)][str(H)]=dict(weighted_fraction=weighted,unweighted_valid=int(valid.sum()),unweighted_total=valid.size)
    js(HERE/'FILTR_NEARZERO.json',filters)
    # Fixed image grid: no spatial movement implied by projected latent trajectories.
    im=image(HERE/'SONDA_STALA.png');fig,ax=plt.subplots(figsize=(13,8.7));ax.imshow(im)
    for j,k in enumerate(ids):
        x,y=(k%30)*16,(k//30)*16;ax.add_patch(Rectangle((x,y),16,16,fill=False,edgecolor='yellow',lw=.9));ax.text(x+1,y+11,str(k+1),fontsize=6,color='black',bbox=dict(facecolor='yellow',alpha=.8,pad=.3))
    for j,k in enumerate(chosen):
        ti=ids[k];ax.add_patch(Rectangle(((ti%30)*16,(ti//30)*16),16,16,fill=False,edgecolor=colors[j],lw=3));ax.annotate(f'{ti+1} [od 0: {ti}]',((ti%30)*16+8,(ti//30)*16+8),xytext=(14,(-25 if j%2 else 25)),textcoords='offset points',color=colors[j],fontsize=10,bbox=dict(facecolor='white',alpha=.8,pad=2))
    ax.set_title(f'Stały obraz: {data["frame_id"]}\n31 cząsteczek; żółte numery 1-based; wyróżniona granica/interior/mediana coverage');ax.axis('off');fig.tight_layout();fig.savefig(HERE/'CZASTECZKI_NA_STALYM_OBRAZIE.png',dpi=135);plt.close(fig)
    with (HERE/'CZASTECZKI.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['token0','token1','row','column','coverage','weight','example_role'])
        for j,k in enumerate(ids):writer.writerow([k,k+1,k//30,k%30,coverage[j],w[j],'boundary' if j==boundary else 'interior' if j==interior else 'median' if j==middle else ''])
    fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for dim,ax in enumerate(axs):
        for j,k in enumerate(chosen):ax.plot(np.arange(1001),q[:,k,dim],color=colors[j],label=f'cząsteczka {ids[k]+1}, coverage={coverage[k]:.3f}',lw=.8)
        ax.axvline(500,color='gray',ls='--');ax.axvline(700,color='orange',ls=':');ax.axvline(800,color='red',ls=':');ax.set_ylabel(f'q{dim+1} = U500[:,{dim}]ᵀz')
    axs[0].legend(ncol=3,fontsize=8);axs[-1].set_xlabel('Aktualizacja ViT / BN');fig.suptitle('Współrzędne cząsteczek w zamrożonych kierunkach U500');fig.tight_layout();fig.savefig(HERE/'RUCH_KIERUNKI_K500.png',dpi=135);plt.close(fig)
    fig,axs=plt.subplots(5,1,figsize=(12,9),sharex=True)
    for dim,ax in enumerate(axs):
        for j,k in enumerate(chosen):ax.plot(a['axes500_values'][:,k,dim],color=colors[j],lw=.8,label=f'cząsteczka {ids[k]+1}')
        ax.set_ylabel(f'e{int(a["axes500_0"][dim])+1}');ax.axvline(500,color='gray',ls='--');ax.axvline(800,color='red',ls=':')
    axs[0].legend(ncol=3);axs[-1].set_xlabel('Aktualizacja');fig.suptitle('Pięć osi wybranych z identyfikacji train na k500: mikroruch składowych');fig.tight_layout();fig.savefig(HERE/'RUCH_OSIE_K500.png',dpi=125);plt.close(fig)
    pred=dict(np.load(HERE/'PROGNOZY_DT1.npz'));fig,axs=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for row,H in enumerate((1,10,50)):
        for col,k in enumerate((boundary,interior)):
            ax=axs[row,col];actual=q[origins+H,k,0]-q[origins,k,0];forecast=pred[f'selected_H{H}'][:,k,0]-q[origins,k,0]
            ax.plot(origins,actual,color='black',lw=.85,label='rzeczywista przyszła zmiana');ax.plot(origins,forecast,color='#b63839',lw=.85,label='autonomiczny GLE');ax.axhline(0,color='gray',ls='--',label='brak zmiany');ax.set_title(f'H={H}, cząsteczka {ids[k]+1}, '+('granica' if col==0 else 'interior'));ax.set_ylabel('Δq1')
    axs[0,0].legend(fontsize=8);axs[-1,0].set_xlabel('Origin (aktualizacja dostępnej historii)');axs[-1,1].set_xlabel('Origin');fig.suptitle('Prognozowanie zmian: H1/10/50, stałe cząsteczki, bez teacher forcing');fig.tight_layout();fig.savefig(HERE/'PROGNOZY_H1_H10_H50.png',dpi=135);plt.close(fig)
    hs=np.array([1,2,5,10,20,50]);fig,axs=plt.subplots(1,3,figsize=(15,4.8))
    for name,style in [('selected','-'),('best_H1','--'),('L0',':'),('velocity','-.')]:
        metric=metrics[1]['models'][name];axs[0].plot(hs,[metric[str(h)]['MSE'] for h in hs],style,marker='o',label=name);axs[1].plot(hs,[100*metric[str(h)]['skill'] for h in hs],style,marker='o',label=name)
    axs[0].plot(hs,[metrics[1]['models']['persistence'][str(h)]['MSE'] for h in hs],color='gray',label='persistence');axs[0].set_yscale('log');axs[0].set_title('MSE w fizycznych q');axs[1].set_ylim(-100,100);axs[1].axhline(0,color='gray');axs[1].set_title('Skill względem braku zmiany [%]')
    main=metrics[1]['models']['selected'];axs[2].plot(hs,[100*main[str(h)]['sign_accuracy'] for h in hs],marker='o',label='znak wszystkich');axs[2].plot(hs,[100*main[str(h)]['sign_filtered'] for h in hs],marker='o',label='znak po filtrze');axs[2].plot(hs,[100*main[str(h)]['majority_sign_accuracy'] for h in hs],ls='--',label='większość train');axs[2].axhline(80,color='gray',ls=':');axs[2].set_title('Znaki współrzędnych, nie całych wektorów [%]')
    for ax in axs:ax.set_xscale('log');ax.set_xlabel('H [aktualizacje]');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'HORYZONTY_DT1.png',dpi=135);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));h10=[10,20,50]
    for name in ('selected','L0'):
        ax.plot(h10,[100*metrics[10]['models'][name][str(h)]['skill'] for h in h10],marker='o',label=f'dt10 {name}')
    ax.plot(h10,[100*metrics[1]['common_co10_origins']['selected'][str(h)]['skill'] for h in h10],marker='o',label='dt1 selected, te same co 10 originy');ax.axhline(0,color='gray');ax.legend();ax.set_xlabel('H [aktualizacje]');ax.set_ylabel('Skill [%]');fig.tight_layout();fig.savefig(HERE/'CHECKPOINTY_CO10.png',dpi=135);plt.close(fig)
    shared=[r for r in noise[1]['per_series_diagnostics'] if r['series']=='shared'];fig,axs=plt.subplots(1,2,figsize=(11,4));res=np.load(HERE/'RESZTY_DT1.npz')['residuals'];shared_e=np.einsum('tnd,n->td',res,w)
    axs[0].plot(np.arange(801,1001),shared_e[:,1],lw=.8);axs[0].set_title('Wspólna reszta one-step, współrzędna q2');axs[0].set_xlabel('Target aktualizacji');axs[1].stem(np.arange(1,11),shared[1]['acf']);axs[1].axhline(0,color='gray');axs[1].set_title('ACF w czasie; gaussowskość badana osobno');fig.tight_layout();fig.savefig(HERE/'RESZTY_ACF.png',dpi=135);plt.close(fig)
    # Retrospective trajectories are deliberately separate, without forecast scores.
    fig,ax=plt.subplots(figsize=(10,4))
    for j,k in enumerate(chosen):ax.plot(retro['q1000'][:,k,0],lw=.7,color=colors[j],label=f'cząsteczka {ids[k]+1}')
    ax.set_title('TYLKO RETROSPEKTYWNIE: U1000 z końca trajektorii');ax.set_xlabel('Aktualizacja');ax.set_ylabel('q1000 kierunek 1');ax.legend();fig.tight_layout();fig.savefig(HERE/'RETROSPEKTYWNIE_U1000.png',dpi=125);plt.close(fig)
    now=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat();m=metrics[1]['models']['selected'];chosen1=metrics[1]['selection']['selected'];chosen10=metrics[10]['selection']['selected'];mod=params[1]['selected'];near=filters['1']['1']
    lines=['# Langevin/GLE ruchu cząsteczek na kierunkach identyfikacji',f'\nUkończono {now}. Jeden stały obraz, 31 odrębnych tokenów, 1001 stanów (0..1000), czyli 1000 aktualizacji ViT/BatchNorm. To dynamika podczas treningu, odrębna od wcześniejszej reakcji na usunięcie narzędzia.',
        f'\n**Główny wynik:** autonomiczny GLE z L={chosen1["L"]}, ridge={chosen1["ridge"]} wybrany na walidacji daje na H1 skill **{100*m["1"]["skill"]:.2f} procent**, znak wszystkich składowych **{100*m["1"]["sign_accuracy"]:.2f} procent**, po odrzuceniu bardzo małych rzeczywistych zmian, z progiem wyznaczonym na fit, **{100*m["1"]["sign_filtered"]:.2f} procent**. Na H50 skill jest ujemny: **{100*m["50"]["skill"]:.2f} procent**. Nie osiągnięto 80 procent ani redukcji MSE, ani znaku wszystkich zmian. Filtr osiąga 81.59 procent tylko na H1 i nie jest trafnością całych wektorów.',
        '\n## Obraz, tokeny i przyczynowość kierunków',
        f'\nSonda wybrana przez roota przed wynikami tego zadania: indeks od 0=2, `{data["frame_id"]}`. Faktyczne source role=fit i split=train. Wcześniejsze określenie jako pierwsza holdout sonda było pomyłką, skorygowaną przed testem. Nie zmieniliśmy obrazu po obejrzeniu wyników. Bieg seed47 i jego późniejsze stany były analizowane wcześniej: temporalny test jest chronologiczny dla bieżącego algorytmu, ale nie stanowi nowego nietkniętego potwierdzenia ani generalizacji na nowy obraz.',
        '\n![Stały obraz i śledzone tokeny](CZASTECZKI_NA_STALYM_OBRAZIE.png)',
        '\nToken i to pełny z_i∈R^768; mikroruch to zmiana współrzędnych względem wspólnej bazy e_j. Maska oryginału, coverage i indeksy 31 tokenów są stałe. Wykresy pokazują indywidualne cząsteczki; pooling jest wyłącznie pomocniczy. CZASTECZKI.csv podaje numery od 0 i od 1, row/column, coverage i wagę. Granicę/interior wybrano według min/max coverage, bez oglądania trajektorii.',
        '\nU500 (768×8) wyznaczono na checkpointcie500 wyłącznie z 240 par train i wagami film→obraz→coverage. Rank 8 zadeklarowano wcześniej. Pozostałe pary nie wpływały na bazę lub ranking osi. Pięć osi wybranych na train500: **359,594,472,512,138** (od 0:358,593,471,511,137).',
        f'\nDiagnostyczny probe C1, próg0.5: identyfikacja na 85 parach testowych AUROC {100*identity["identity_evaluation"]["test"]["AUROC"]:.2f} procent, balanced accuracy {100*identity["identity_evaluation"]["test"]["balanced_accuracy_threshold0_5"]:.2f} procent. To **image-weighted**, znany ROI oryginał/LaMa; nie porównujemy bezpośrednio z wcześniejszym film-weighted U1000/progiem wybieranym na val.',
        f'\nLokalny evalforward k500 na dokładnym obrazy.npy[2] zgadza się z LATENT_GESTY[500,2]: relativeL2 {alignment["relative_l2_error"]:.3g}, maxabs {alignment["max_abs_error"]:.3g}, cosine {alignment["cosine"]:.15f}. Tolerancja relative1e-4, pełne 600×768. U500 i zapisane kanały są zgodne numerycznie. Źródłowy pomiar A100/PyTorch2.9 versus lokalny RTX3070/PyTorch2.6 daje drobne różnice float32.',
        '\nU1000 oraz poprzednie osie 500/472/735/512/647 są wyłącznie opisem retrospektywnym, w osobnym NPZ. Nie użyto ich do prognozy. Root wyznaczył kąty główne U500/U1000 w porownanie_podprzestrzeni/: 27.69,30.56,31.41,35.54,42.72,44.10,58.60,83.95 stopnia; mean cos²=.531748. To geometra zmiany dwóch ponownie identyfikowanych baz, bez zmiany cech tego testu. Nie interpretujemy U jako stałej uniwersalnej tożsamości.',
        '\n![Ruch tokenów w U500](RUCH_KIERUNKI_K500.png)\n\nPionowe linie: 500 oznacza checkpoint identyfikacji U500 i anchor; 700 koniec pierwotnego fit/początek walidacji; 800 początek originów testowych. Po wyborze modelu końcowy refit obejmował cele do 799.', '\n![Mikroruch pięciu osi500](RUCH_OSIE_K500.png)',
        '\n## Równanie, estymacja i jednostki',
        '\n```text\nq_i(t)=U500ᵀ z_i(t),  x_i(t)=q_i(t)−q_i(500)\nΔx_i(t)=x_i(t)−x_i(t−dt)\nx_i(t+dt)=x_i(t)+b+A x_i(t)\n            +Σ_(ell=0..L−1) K_ell Δx_i(t−ell*dt)+epsilon_i(t+dt)\n```',
        '\nb,A,K są wspólne dla tokenów i estymowane JEDNOCZEŚNIE ridge, z równymi wagami targetów i coverage-weighted tokenami. Stały tokenowy anchor usuwa różnice wyjściowych położeń. W surowym q wyraz wolny jest b_i=b−Aq_i(500). Jest to empiryczny GLE/AR z pamięcią i skończona affine aproksymacja Koopmana na obserwablach q plus stała. Nie dowodzi dokładnego Mori–Zwanziga, globalnej liniowości ani Brownowskiego szumu.',
        '\nŹródłowe wejścia treningu: paczka32, AdamW lr=.0005, weight_decay=.05, warmup300. Bieg nie zapisywał faktycznych batch-ID/augmentacji/gradientów każdej aktualizacji. Dlatego ten model **nie ma jawnego członu sterowania**; nie przeniesiono telemetrii innych biegów. Ruch w eval obejmuje zmiany wag i biegnących statystyk BatchNorm; nie przypisujemy go wyłącznie gradientowi ani semantyce.',
        '\nSkale stanu s_x i przyrostu s_d, intercept i ridge pochodzą tylko z fit. NumPy używa wierszy F@B; równanie wyżej używa kolumn. Jeśli B_x i B_lag są blokami fitted slope, macierze w fizycznych współrzędnych to:',
        '\n```text\nb = diag(s_d) intercept\nA = diag(s_d) B_xᵀ diag(1/s_x)\nK_ell = diag(s_d) B_lag,ellᵀ diag(1/s_d)\n```',
        f'\nNa wszystkich regresorach fit skalowane i fizyczne równanie są numerycznie identyczne: maks. błąd dla wybranego dt1 {mod["native_vs_scaled_max_error"]:.3g}. b/epsilon mają jednostki aktywacji na krok dt; A,K są bezwymiarowe dla tej dyskretyzacji. W q macierze stosujemy bez ukrytej normalizacji. To przyrost latentu, nie pęd AdamW. Gram rank {mod["gram_rank"]}/{mod["feature_count"]}, regularized condition {mod["regularized_condition"]:.3g}, companion spectral radius {mod["companion_spectral_radius"]:.6f}. Ridge stabilizuje predykcję, nie dowodzi fizycznej identyfikowalności A/K.',
        '\nMacierze, skale, anchor i próg selected_epsilon do filtrowanej metryki znaku oraz wszystkie współczynniki do autonomicznego rollout zapisane w MACIERZE_DT1/DT10.npz; metadane MODELE_DT*.json. Ridge gamma = lambda*trace(Gram)/liczba_features, Gram z sumą wag 1, intercept niepenalizowany. Lambda 1e-4/1e-3/1e-2/.1/1, L0/1/2/4/8. Żadnego Kalmana. Nazwa selected_epsilon w NPZ oznacza 8D próg near-zero, a nie realizację reszty epsilon_i(t) z równania; reszty są zapisane osobno w RESZTY_DT*.npz.',
        '\n## Podział, wybór i autonomiczny test',
        '\ndt1: fit targets 501..700, wspólne wszystkim L (L8 korzysta z poznanej historii 492..500). Validation origins 700..749; H1/10/50, target<=799. Wybór maximum MEAN skill trzech horyzontów; nie surowego pooledMSE. Po wyborze refit targets <=799. Test origins 800..950, wszystkie modele te same151originów dla H1/2/5/10/20/50. H100 ma osobny zestaw101originów 800..900, bez CI.',
        f'\nWybrane dt1: L{chosen1["L"]}, ridge{chosen1["ridge"]}; validation mean skill {100*chosen1["mean_skill"]:.2f} procent. Osobno zadeklarowany bestH1 to L2/.01, testH1skill {100*metrics[1]["models"]["best_H1"]["1"]["skill"]:.2f} procent. Nie zmieniono głównego modelu po teście.',
        '\ndt10: targets 510..700 co 10, validation origins 700..740 co 10, równoważnie średni skill H10/50; refit do 790. Model L1/ridge1 wybrany przed testem,16originów 800..950 co 10. H1/2/5 niedostępne, nie nazywamy dt10 jedną aktualizacją. Artefakt best_H1 dla tego modelu oznacza bestH10. dt1 dodatkowo oceniono na tych samych co 10 originach.',
        '\nW rollout używano własnych przewidzianych przyrostów; żadnych przyszłych q, paczek, obrazów, wag, gradientów lub reszt. Aktualny stan originu i dostępna historia są wejściem rolling forecast. Osobne testy dokładnego prefixu i zatrutego NaN przyszłego sufiksu dały identyczne prognozy dla wszystkich modeli przy origin 800/950, H50. Skale i parametry pozostały zamrożone. Test wykonano po audycie Astry, jednokrotnie.',
        '\n## Wyniki fizycznego błędu i kierunku',
        '\nMSE = średnia po originach i 8 współrzędnych kwadratu błędu, z coverage-weighted tokenami. Skill=1−MSE/MSEpersistence. Cosine dotyczy wektorów ZMIANY forecast versus actual, nie poziomu. Znak oznacza poprawność pojedynczych współrzędnych. Persistence ma zerową zmianę: cosine nieokreślony, a nie idealny.',
        '\n| H [aktualizacje] | MSE GLE | MSE brak zmiany | Skill | Znak wszystkich | Znak po filtrze | Cosine zmian | Większość znaku train |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for H in (1,2,5,10,20,50,100):
        r=m[str(H)];lines.append(f'|{H}|{r["MSE"]:.6f}|{r["persistence_MSE"]:.6f}|{100*r["skill"]:.2f}%|{100*r["sign_accuracy"]:.2f}%|{100*r["sign_filtered"]:.2f}%|{r["cosine"]:.4f}|{100*r["majority_sign_accuracy"]:.2f}%|')
    lines.extend([f'\nH100 jest oddzielną diagnostyką na innych originach: 101 dla dt1 i 11 dla dt10. Nie jest kolejnym punktem tego samego testowego zbioru ani bezpośrednim porównaniem obu dyskretyzacji; odbicie skilla nie dowodzi poprawy przy dłuższym horyzoncie. H1 validcosine={m["1"]["cosine_valid"]}/{m["1"]["cosine_total"]}, H50={m["50"]["cosine_valid"]}/{m["50"]["cosine_total"]}. Wektorowa cosine H1=.74024 nie oznacza 74procent poprawnych całych wektorów.',
        f'\nFiltr: epsilon_j=.1×ważona SD przyrostów fit do 799. Stały dla wszystkich H. Odsiew abs(rzeczywista przyszła zmiana)>epsilon jest wykonywany dopiero podczas oceny po poznaniu celu. To warunkowa metryka, nie dostępna online bramka pewności prognozy. H1 zachowuje **{near["unweighted_valid"]}/{near["unweighted_total"]}** porównań składowych ({100*near["unweighted_valid"]/near["unweighted_total"]:.2f}% bez ważenia; {100*near["weighted_fraction"]:.2f}% ważone coverage). 81.59 procent dotyczy tej przefiltrowanej metryki, nie wszystkich zmian i nie całych wektorów. Maksymalny badany horyzont osiągający 80procent tej metryki toH1; cały znak i skill nie osiągają 80 na żadnym badanym H.',
        '\n| Ramię dt1 | H1 skill | H10 skill | H50 skill |','|---|---:|---:|---:|'])
    for name in ('L0','L1','L2','L4','L8','selected','best_H1','velocity','persistence'):
        r=metrics[1]['models'][name];lines.append(f'|{name}|{100*r["1"]["skill"]:.2f}%|{100*r["10"]["skill"]:.2f}%|{100*r["50"]["skill"]:.2f}%|')
    lines.extend(['\n![Autonomiczne prognozy indywidualnych tokenów](PROGNOZY_H1_H10_H50.png)', '\n![Horyzont i błędy](HORYZONTY_DT1.png)',
        '\n## Checkpoint co 10, uczciwe porównanie originów',
        '\n| H | dt1 selected skill, co 10 originy | dt10 L0 skill | dt10 selected L1 skill | dt10 selected znak |','|---:|---:|---:|---:|---:|'])
    for H in (10,20,50):
        r=metrics[10]['models']['selected'][str(H)];lines.append(f'|{H}|{100*metrics[1]["common_co10_origins"]["selected"][str(H)]["skill"]:.2f}%|{100*metrics[10]["models"]["L0"][str(H)]["skill"]:.2f}%|{100*r["skill"]:.2f}%|{100*r["sign_accuracy"]:.2f}%|')
    lines.extend(['\n![Modele na wspólnych co 10 originach](CHECKPOINTY_CO10.png)',
        '\nPamięć wnosi główną poprawę na H1/dt1: Markov jest blisko braku zmiany, a velocity daje 47.77procent skilla wobec 60.24GLE. Przy dt10/H10 L0 jest niemal równie dobry jakL1; nie wykazujemy przewagi pamięci w każdej skali.',
        '\n## Niepewność i kontrola wspólnej dynamiki',
        '\nCircular moving block bootstrap 300, blok 50aktualizacji, wspólne losowanie całego kompletu tokenów/modeli; dt10 blok 5originów. Tylko około 3efektywne bloki, więc CI są **opisowe**, nie niezależne potwierdzenie z wieloma ziarnami. H100 nie ma CI (około 1blok). Bieg i obraz były wcześniej analizowane.',
        f'\n95% CI dt1: H1skill{interval(m["1"]["bootstrap"]["CI95"]["skill"])}%, znak wszystkich {interval(m["1"]["bootstrap"]["CI95"]["sign_accuracy"])}%, po filtrze{interval(m["1"]["bootstrap"]["CI95"]["sign_filtered"])}%. H10skill{interval(m["10"]["bootstrap"]["CI95"]["skill"])}%; H50skill{interval(m["50"]["bootstrap"]["CI95"]["skill"])}%. dt10H50punktowo 13.28%, CI{interval(metrics[10]["models"]["selected"]["50"]["bootstrap"]["CI95"]["skill"])}%, obejmuje 0. Brak gwarancji monotonicznego wyniku z H.',
        f'\nTransfer tego samego modelu do8stałych farbackgroundtokens (zmieniono tylko anchor, operator bez refitu): H1skill{100*metrics[1]["background_transfer"]["1"]["skill"]:.2f}%, znak {100*metrics[1]["background_transfer"]["1"]["sign_accuracy"]:.2f}%. Jest to zbliżone do tooltokens, więc przewidywalność nie jest wyjątkowa dla narzędzia. H10/H50 background skill: {100*metrics[1]["background_transfer"]["10"]["skill"]:.2f}/{100*metrics[1]["background_transfer"]["50"]["skill"]:.2f}%. Model opisuje także wspólne zmiany sieci/BN; nie rozstrzygamy przyczyny.',
        '\n## Reszta: kolor, kształt rozkładu i niedomknięcie',
        '\nReszta one-step z ciągłego target801..1000 (200próbek) używa rzeczywistej historii do kolejnego originu; jest osobna od autonomicznych błędów H. ACF/Ljung–Box badane w czasie per token / per coordinate i dla shared residual. Gaussowskość osobno przez skew/kurtosis/Jarque–Bera. Nie spłaszczono tokenów do fałszywej jednej serii.',
        f'\nRaw Ljung–Box p<.05 dla {sum(r["LjungBox_raw_p"]<.05 for r in noise[1]["per_series_diagnostics"])}/256szeregów; Jarque–Bera p<.05 dla {sum(r["JarqueBera_p"]<.05 for r in noise[1]["per_series_diagnostics"])}/256. Bez korekty wielokrotnych testów i degreesoffreedom dopasowanych parametrów: diagnostyczne wyniki, nie certyfikat white/Gaussian. Dla shared q2 JB p={shared[1]["JarqueBera_p"]:.4f} (nie odrzuca), Ljung–Box p={shared[1]["LjungBox_raw_p"]:.3g} (korelacje); brak odrzucenia gaussowskości nie implikuje białości.',
        '\nCross-token mediany korelacji reszt w tej samej współrzędnej: '+ ', '.join(f'{r["median"]:.3f}' for r in noise[1]['cross_token_same_coordinate_correlations'])+'. Są dodatnie i duże: reszta zawiera wspólny składnik aktualizacji sieci. Zmiany wariancji połów i pełne ACF w SZUM_DT*.json. Nie narzucamy FDT dla AdamW ani fizycznego white Brownian noise; niedomknięcie pozostaje.',
        '\n![Reszty i ACF](RESZTY_ACF.png)',
        '\n## Co uzyskano i czego nie',
        '\nWyznaczono przyczynowo dostępne kierunki 500, pełne indywidualne trajektorie 31tokenów i wspólny empiryczny GLE z pamięcią. Poprawiono krótką prognozę względem persistence/velocity/Markov; nie wykazano 80procent na wszystkich zmianach, dobrego H50dla głównego modelu, wyjątkowej dynamiki semantycznej narzędzia ani dokładnego równania fizycznego. Brak telemetrii minibatchy ogranicza interpretację dryfu/reszt. Potrzebne nowe niezależne biegi i obrazy oraz jawne historyczne wejścia, bez prawdziwych przyszłych paczek.',
        '\n## Artefakty i odtwarzanie',
        '\nPLAN.md; IDENTYFIKACJA_K500.npz/json (U500); TRAJEKTORIE_PRZYCZYNOWE_K500.npz; osobno TRAJEKTORIE_RETROSPEKTYWNE_K1000.npz; MACIERZE_DT1/DT10.npz; WYBOR_WALIDACJA_DT*.json; METRYKI_DT*.json; PROGNOZY_DT*.npz; SZUM/RESZTY_DT*; AUDYT_PREFIX_DT*; ZGODNOSC_K500.json; CZASTECZKI.csv; AUDYT_ASTRA.md. Root ma własność porownanie_podprzestrzeni/ i PODSUMOWANIE.md.',
        '\nF:/Python/python.exe, kolejno identyfikacja500.py, trajektorie.py, prognoza.py --stage validation; audit Astry przed prognoza.py --stage test; raport.py. Bez kopiowania 22 GB, pełne wektory 768D przez mmap tylko w RAM, małe NPZ. Bez pobierania, treningu i commitów.'])
    # Minimal whitespace cleanup in generated Polish prose (preserve code/hash/filenames).
    replacements={'checkpointcie500':'checkpointcie 500','train500':'train 500','500/':'500 /','31tokenów':'31 tokenów','31tok':'31 tok','dt1/H':'dt1 / H','dt10/H':'dt10 / H','dt10H':'dt10 H','H1skill':'H1 skill','H10skill':'H10 skill','H50skill':'H50 skill','0.5:':'0.5:','image-weighted':'image-weighted','próg0.5':'próg 0.5','probe C1':'probe C=1','relative1e-4':'relative 1e-4','full600':'full 600','rank8':'rank 8','Role=':'Role =','degreesoffreedom':'degrees of freedom','source role=fit':'source role = fit','split=train':'split = train','target801':'target 801','200próbek':'200 próbek','/256szeregów':'/256 szeregów','50aktualizacji':'50 aktualizacji','300,':'300,','5originów':'5 originów','3efektywne':'3 efektywne','1blok':'1 blok','przyszrostach':'przyrostach','per token':'per token','wyjątkowej dynamiki':'wyjątkowej dynamiki','train-only':'train-only','H50dla':'H50 dla','zero0':'zero 0','80procent':'80 procent','81.59 procent':'81.59 procent','74procent':'74 procent','rank8':'rank 8','47.77procent':'47.77 procent','60.24GLE':'60.24 GLE','Markov jest':'Markov jest','toH1':'to H1','przydt10':'przy dt10','jakL1':'jak L1','samych151originów':'samych 151 originów','101originów':'101 originów','16originów':'16 originów','do8stałych':'do 8 stałych','time200':'time 200','0.01;':'0.01;','geometra':'geometria'}
    text='\n'.join(lines)+'\n'
    for old,new in replacements.items():text=text.replace(old,new)
    (HERE/'WYNIKI.md').write_text(text,encoding='utf-8')
    (HERE/'README.md').write_text('# GLE ruchu cząsteczek: kierunki U500\n\nZakończona eksploracyjna analiza stałego obrazu, 31 tokenów i 1001 stanów (0..1000), czyli 1000 aktualizacji. [Raport](WYNIKI.md), [podsumowanie roota](PODSUMOWANIE.md), [plan](PLAN.md), [audyt Astry](AUDYT_ASTRA.md). Kierunki500 wyznaczono wyłącznie train; U1000 tylko retrospektywnie.\n\nOdtwarzanie z F:/Python/python.exe w tym katalogu: identyfikacja500.py, trajektorie.py, prognoza.py --stage validation, audyt, prognoza.py --stage test, raport.py. Nie nadpisuje źródeł. Małe NPZ zawierają parametry/projekcje/predykcje; źródłowe 22 GB używa mmap. Rola sondy 2 = fit/train, nie niezależny obraz. Brak logów minibatchy, bez nowego treningu i commitów.\n',encoding='utf-8')
    total=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());free=shutil.disk_usage(HERE).free;assert total<150e6 and free>1e9
    js(HERE/'WYNIKI.json',dict(family='maly_vit_langevin_kierunki_tozsamosci',status='complete_exploratory',created=now,probe=data,identification500=identity,alignment=alignment,selected_metrics={str(dt):metrics[dt]['models']['selected'] for dt in (1,10)},nearzero_filter=filters,
        validation_choices={str(dt):metrics[dt]['selection'] for dt in (1,10)},code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},bytes_total=total,disk_free=free,source_arrays_not_copied=True,training=False,new_downloads=False))
    print('Report ready',total,'bytes, free',free,flush=True)

if __name__=='__main__':main()
