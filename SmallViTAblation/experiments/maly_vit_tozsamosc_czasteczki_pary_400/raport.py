"""Raport odróżnia reakcję i wpływ; brak sham/negatywów obala pełną interpretację."""
import json,time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dane import HERE,DATA,load,image,weights,js,sha
from identyfikacja import boot_presence

def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))

def main():
    rows,t=load();measurement=read('POMIAR.json');fit=read('IDENTYFIKACJA.json');det=read('DETEKCJA.json');axes=read('MAKROWEKTORY.json');token=read('CZASTECZKI_RANKING.json');individual=read('CZASTECZKI_WPLYW.json')
    cache=dict(np.load(HERE/'TOKENY_PARY.npz'));basis=dict(np.load(HERE/'PODPRZESTRZEN.npz'));records=read('DETEKCJA_OBRAZY.json');test=np.array([i for i,r in enumerate(rows) if r['split']=='test']);w=weights(rows,test)
    pooled=cache['pooled'];lda=basis['lda_direction'];center=basis['lda_center'];scores=np.r_[(pooled[test,0]-center)@lda,(pooled[test,1]-center)@lda]
    fit['lda']['film_bootstrap95']=boot_presence(rows,test,scores,float(basis['lda_threshold']))
    js(HERE/'IDENTYFIKACJA.json',fit)
    # Paired identity minus mean of all three prespecified random controls.
    rec={m:[r for r in records if r['method']==m] for m in det['summary']}
    q_identity=np.array([r['q'] for r in rec['identity']]);q_random=np.mean([[r['q'] for r in rec[f'random{s}']] for s in (11,22,33)],axis=0)
    film_deltas=np.array([(q_identity-q_random)[[j for j,i in enumerate(test) if rows[i]['video_id']==f]].mean() for f in sorted({rows[i]['video_id'] for i in test})])
    rng=np.random.default_rng(912);b=np.array([film_deltas[rng.integers(0,len(film_deltas),len(film_deltas))].mean() for _ in range(500)])
    additional=dict(identity_minus_mean_random_q=float(w@(q_identity-q_random)),film_bootstrap95=list(np.quantile(b,[.025,.975])),effective_norms={m:dict(hidden=float(w@np.array([r.get('hidden_norm',0) for r in rec[m]])),fc2=float(w@np.array([r.get('functional_fc2_norm',0) for r in rec[m]]))) for m in det['summary']})
    js(HERE/'KONTROLE_SELEKTYWNOSCI.json',additional)
    shams={tag:read(f'SHAM_DIAGNOSTYKA_{tag}.json') for tag in ('mixed','heldout')}
    summaries={}
    for tag,ss in shams.items():
        ids=np.array([r['source_index'] for r in ss]);sw=weights(rows,ids)
        means={k:float(sw@np.array([r[k] for r in ss])) for k in ('original_tool_roi_delta_l2','original_tool_roi_projection_l2','sham_roi_delta_l2','sham_roi_projection_l2','probability_delta')}
        # Same source images: removal tool ROI compared with sham original tool ROI.
        proj=read('PROJEKCJA_TLO.json');means.update(removal_tool_projection_mean=float(sw@np.array([proj[i]['tool_projection_mean'] for i in ids])),removal_tool_delta_mean=float(sw@np.array([proj[i]['tool_delta_mean'] for i in ids])),
            presence_call_flips=sum(r['original_presence_call']!=r['sham_presence_call'] for r in ss),n=len(ss),split_counts={s:sum(r['split']==s for r in ss) for s in ('train','val','test')},nominal_roi_overlap_total=sum(r['nominal_token_roi_overlap'] for r in ss))
        summaries[tag]=means
    js(HERE/'SHAM_PODSUMOWANIE.json',summaries)
    background=read('PROJEKCJA_TLO.json');bgtest={k:float(w@np.array([background[i][k] for i in test])) for k in ('tool_projection_mean','far8_projection_mean','tool_delta_mean','far8_delta_mean')}
    js(HERE/'TLO_TEST_PODSUMOWANIE.json',bgtest)
    # Rankings preserve both channel conventions and signed effects.
    fig,ax=plt.subplots(figsize=(10,7));top=axes[:20];values=np.array([r['paired_effect'] for r in top]);ax.barh(range(20),values,color=np.where(values>=0,'#1a7f75','#b24158'));ax.set_yticks(range(20),[f'e{r["channel1"]} (index0={r["channel0"]})' for r in top]);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Train: weighted paired mean delta / SD');ax.set_title('Najbardziej spójne reagujące składowe; ranking nie jest dowodem wpływu');fig.tight_layout();fig.savefig(HERE/'MAKROWEKTORY_TOP20.png',dpi=135);plt.close(fig)
    eig=basis['eigenvalues'];fig,ax=plt.subplots(figsize=(8,4));ax.plot(np.arange(1,769),np.cumsum(eig)/eig.sum());ax.axhline(.5,color='gray',ls='--');ax.axhline(.9,color='gray',ls='--');ax.axvline(fit['chosen_rank'],color='red',ls='--',label=f'wybrane r={fit["chosen_rank"]}');ax.set_xlim(1,150);ax.set_ylim(0,1);ax.set_xlabel('Liczba kierunków');ax.set_ylabel('Energia train delta');ax.legend();fig.tight_layout();fig.savefig(HERE/'ENERGIA_PODPRZESTRZENI.png',dpi=130);plt.close(fig)
    example_descriptions=[]
    for ex in individual:
        i=ex['source_index'];r=rows[i];original=image(DATA/r['image_path']);negative=image(r['synthetic_absolute']);reaction=token[i]['tokens'][0];influence=ex['token_influence'][0]
        influence_map=np.zeros(600)
        for q in ex['token_influence']:influence_map[q['token0']]=q['delta_q']
        fig,axs=plt.subplots(2,3,figsize=(13,6.4))
        for ax in axs.flat:ax.axis('off')
        axs[0,0].imshow(original);axs[0,0].set_title('Oryginał; GT bbox i maska tokenów')
        bb=r['bbox_source_480'];axs[0,0].add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],fill=False,edgecolor='lime',lw=1.4))
        axs[0,0].contour(np.repeat(np.repeat(t['tool'][i].reshape(20,30),16,axis=0),16,axis=1),levels=[.5],colors='yellow',linewidths=.6)
        axs[0,1].imshow(negative);axs[0,1].set_title('LaMa; syntetyczne tło')
        axs[0,2].imshow(original,alpha=.45);p=axs[0,2].imshow(cache['full_delta_l2'][i].reshape(20,30),extent=(0,480,320,0),alpha=.65,cmap='magma',interpolation='nearest');axs[0,2].set_title('L2 zmiany pełnych tokenów (cała scena)');fig.colorbar(p,ax=axs[0,2],fraction=.04)
        axs[1,0].imshow(original,alpha=.45);p=axs[1,0].imshow(basis['identity_projection_maps'][i].reshape(20,30),extent=(0,480,320,0),alpha=.65,cmap='magma',interpolation='nearest');axs[1,0].set_title('Projekcja na wybrane U; tylko tool ROI');fig.colorbar(p,ax=axs[1,0],fraction=.04)
        rx,ry=16*reaction['column']+8,16*reaction['row']+8
        axs[1,0].scatter([rx],[ry],facecolors='none',edgecolors='cyan',s=110,lw=1.5)
        axs[1,0].annotate(f'{reaction["token1"]} [od0: {reaction["token0"]}]',(rx,ry),xytext=(8,-23),textcoords='offset points',color='cyan',fontsize=9,bbox=dict(facecolor='black',alpha=.7,pad=2))
        lim=max(np.abs(influence_map).max(),1e-9);axs[1,1].imshow(original,alpha=.4);p=axs[1,1].imshow(influence_map.reshape(20,30),extent=(0,480,320,0),alpha=.65,cmap='coolwarm',vmin=-lim,vmax=lim,interpolation='nearest');axs[1,1].set_title('Zmiana confidence×IoU: jeden token');fig.colorbar(p,ax=axs[1,1],fraction=.04)
        ix,iy=16*influence['column']+8,16*influence['row']+8
        axs[1,1].scatter([ix],[iy],facecolors='none',edgecolors='lime',s=110,lw=1.5)
        axs[1,1].annotate(f'{influence["token1"]} [od0: {influence["token0"]}]',(ix,iy),xytext=(8,22),textcoords='offset points',color='lime',fontsize=9,bbox=dict(facecolor='black',alpha=.7,pad=2))
        axs[1,2].text(0,1,f'{r["video_id"]}\nframe {r["frame_index"]}\nNative axis: {r["angle_axis_deg"]:.1f}°\nInput axis: {r["angle_axis_480_deg"]:.1f}°\n\nMax projection: token0={reaction["token0"]}\nrow={reaction["row"]}, col={reaction["column"]}\ncoverage={reaction["coverage"]:.3f}\nL2={reaction["delta_l2"]:.3f}, proj={reaction["identity_projection_l2"]:.3f}\n\nMax |delta q|: token0={influence["token0"]}\nrow={influence["row"]}, col={influence["column"]}\ndelta q={influence["delta_q"]:+.5f}\ndelta loss={influence["delta_loss"]:+.5f}',va='top',fontsize=10)
        fig.suptitle(f'Test: source_index={i}. Reakcja po usunięciu narzędzia, nie ruch treningowy.');fig.tight_layout();filename=f'HEATMAP_{i:03d}.png';fig.savefig(HERE/filename,dpi=125);plt.close(fig)
        example_descriptions.append(dict(source_index=i,frame_id=r['frame_id'],figure=filename,largest_projection=reaction,largest_metric_change=influence))
    js(HERE/'PRZYKLADY_OPIS.json',example_descriptions)
    angle=read('KATY.json');changedbins=sum(int(r['angle_axis_deg']//15)!=int(r['angle_axis_480_deg']//15) for r in rows)
    now=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat();base=det['summary']['nohook'];ident=det['summary']['identity'];presence=fit['presence']['test']['film_weighted'];ci=fit['presence']['test']['film_bootstrap95'];sh=summaries['heldout'];bg=bgtest
    lines=[f'# Tożsamość narzędzia: odpowiedź cząsteczek na usunięcie z 400 par',f'\nUkończono {now}. Zamrożony model od zera seed47, checkpoint1000aktualizacji z rozgrzewki. To analiza reakcji na zmianę wejścia, nie trajektoria uczenia.',
    f'\n**Wynik:** wybrane na walidacji8kierunków rozdziela znany ROI oryginał/LaMa na85parach innych24filmów: AUROC{100*presence["auroc"]:.2f}%, balanced accuracy{100*presence["balanced_accuracy"]:.2f}%. Regularizowana LDA768D daje BA{100*fit["lda"]["test"]["balanced_accuracy"]:.2f}%. Nie jest to dowód swoistości wobec innych narzędzi. Interwencja8D obniża confidence×IoU bardziej niż trzy norm-matchedrandom, ale nie wykazuje jednoznacznego spadku AP50 lub wzrostu straty; różna siła po fc2 pozostaje alternatywnym wyjaśnieniem.',
    '\n## Aparat i podział',
    f'\n400par z116filmów: train69filmów/240par, val23/75, test24/85. Filmy, nie tokeny, są jednostką podziału i bootstrapu. Oba człony są razem. Dane źródłowe należą do train detektora; test oznacza odłożenie do identyfikacji podprzestrzeni, nie zewnętrzny heldout detektora. Pacjenci niepotwierdzeni.',
    f'\nPełny obraz resize AREA480×320 RGB, normalizacja w modelu. LastblockMLPpoGELU:600×768. FP32eval, BNzamrożone, brakautocast/TF32. Pomiar800obrazów{measurement["seconds"]:.2f}s; zapis6729×2×768 pełnych tokenów maski źródłowej i3200×2×768kontroli tła, losslessfloat32, NPZ{measurement["npz_bytes"]/1e6:.2f}MB. Żadnego średniego wektora nie podstawiono za przechowywane cząsteczki. NoNaN,0pustychROI, identyczny powtórny forward (maxerror0), BNbez zmian. Latenty wszystkich600tokenów nie są zapisane; skrypt je odtwarza, małe mapy energii są zachowane.',
    f'\nModelSHA256 `{measurement["checkpoint_sha256"]}`. Kodmodelu `{measurement["model_source_sha256"]}`. Dane/manifest `{measurement["manifest_sha256"]}`. NPZ `{measurement["npz_sha256"]}`. Revision/hashesLaMa i maskaQC1pary w źródłowym raporcie. RuntimePython{measurement["versions"]["python"]}, PyTorch{measurement["versions"]["torch"]}, timm{measurement["versions"]["timm"]}, RTX3070; bez treningu/Eden/commitów.',
    '\n## Które makrowektory reagują',
    '\nWspólna baza to768osi e_j, nie768 osobnych konceptów. Poniższe składowe mają największą spójną paired odpowiedź średnich ważonych coverage, ranking ustalony wyłącznie train. Delta=oryginał minusLaMa. Znak minus oznacza większą aktywację po edycji. Std jest rozrzutem między obrazami; jednostkowa aktywacja jest w arbitralnej skali sieci. To ranking reakcji, nie bezpośredni pomiar odpowiedzialności pojedynczej osi.',
    '\n| Oś e_j (1-based) | channel0 | Train mean delta | Train SD | mean/SD | Zgodność znaku filmów train | Zgodność znaku obrazów test |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in axes[:10]:lines.append(f'|{r["channel1"]}|{r["channel0"]}|{r["train_mean_delta"]:+.5f}|{r["train_std_delta"]:.5f}|{r["paired_effect"]:+.3f}|{100*r["train_film_sign_agreement"]:.1f}%|{100*r["test_image_sign_agreement"]:.1f}%|')
    lines.extend(['\n![Reagujące osie](MAKROWEKTORY_TOP20.png)',f'\nSVD niescentrowanego drugiego momentu train delty z wagami film→obraz→coverage: r50={fit["r50"]}, r90={fit["r90"]}. Wybrane rank8wyjaśnia{100*fit["energy_selected_rank"]:.2f}%energii delty. Energetyczne r90nie jest liczbą semantycznych kierunków. Rank8/C{fit["chosen_C"]} wybrano valAUROC z20kombinacji; pełna siatka w IDENTYFIKACJA.json. Próg{fit["chosen_threshold"]:.6f}wybrano na val; nie strojono testu. 0degenerującychstdosi.',
    '\nKierunki są kombinacjami wielu makrowektorów. PełneU w PODPRZESTRZEN.npz; poniżej pięć największych wag każdego wybranego kierunku (orientacja znaku na train).',
    '\n| Kierunek | Największe wagi e_j (channel0) |','|---:|---|'])
    for r in fit['mixed_directions']:lines.append('|'+str(r['direction'])+'|'+'; '.join(f'e{v["channel1"]} ({v["channel0"]}): {v["weight"]:+.3f}' for v in r['top_channels'][:5])+'|')
    ldatop=np.argsort(np.abs(lda))[::-1][:10];lines.append('\nLDA ridge1, osobny comparator dyskryminacyjny: największe wagi '+ '; '.join(f'e{j+1} (channel0={j}):{lda[j]:+.4f}' for j in ldatop)+'. LDA kierunek klasyfikacji nie jest automatycznie kierunkiem zapisu informacji.')
    lines.extend(['\n## Rozdzielenie poza dopasowaniem',
    '\n| Metoda | Val AUROC | Test AUROC | Test balanced accuracy |','|---|---:|---:|---:|',
    f'|SVD8 + logistic C1|{100*fit["presence"]["val"]["film_weighted"]["auroc"]:.2f}%|{100*presence["auroc"]:.2f}%|{100*presence["balanced_accuracy"]:.2f}%|',
    f'|LDA ridge1|{100*fit["lda"]["val_auroc"]:.2f}%|{100*fit["lda"]["test"]["auroc"]:.2f}%|{100*fit["lda"]["test"]["balanced_accuracy"]:.2f}%|',
    f'\nSVD95%filmbootstrap AUROC[{100*ci["auroc"][0]:.2f},{100*ci["auroc"][1]:.2f}]%, BA[{100*ci["balanced_accuracy"][0]:.2f},{100*ci["balanced_accuracy"][1]:.2f}]%. LDA95%BA[{100*fit["lda"]["film_bootstrap95"]["balanced_accuracy"][0]:.2f},{100*fit["lda"]["film_bootstrap95"]["balanced_accuracy"][1]:.2f}]%. Główne miary są film-weighted. Image-weightedSVD BA{100*fit["presence"]["test"]["image_weighted"]["balanced_accuracy"]:.2f}%. Labelznanyoryginał/LaMa i ROIz maski oryginału dostępne w obu członach: nie pełnoobrazowy test detekcji.',
    f'\nKATY.json obejmuje12natywnych binów orientacji i leave-angle-out refit U/probe poza binem. Rank/C odziedziczone z val wszystkich kątów; nie rygorystyczny unseen-anglebenchmark. {changedbins}/400zmienia bin po affine resize480×320; biny nazywamy orientacją natywnej anotacji. Zakres testBA po binach: {100*min(r["balanced_accuracy"] for r in angle["test_bins"]):.1f}–{100*max(r["balanced_accuracy"] for r in angle["test_bins"]):.1f}%; małe liczebności{min(r["n"] for r in angle["test_bins"])}–{max(r["n"] for r in angle["test_bins"])}par.',
    '\n## Wpływ na detekcję, odrębny od reakcji',
    '\n85teSame testowe oryginały i GTxyxy (manifestxywh jawnie przekształcony). HooklastGELU: z_o - UUᵀ(z_o-z_n) na maskowanych tokenach. Pełne patchreplacement i trzy random r8ortogonalne, normy dopasowanePER TOKEN, seeds11/22/33. Badany pairedcounterfactual zna z_n jako człon eksperymentu; nie jest prognozą bez znanego negatywu. Noopidentyczny. Po tej warstwie brak attention:fc2→residual→LN→head są punktowe, choć decode maxpool/topk zależy od sąsiadów.',
    '\n| Ramię | AP50 | Film-weighted loss | Film-weighted confidence×IoU | Hidden norm | Norma po fc2 |','|---|---:|---:|---:|---:|---:|'])
    for name,r in det['summary'].items():
        n=additional['effective_norms'][name];lines.append(f'|{name}|{100*r["AP50"]:.2f}%|{r["loss"]:.5f}|{r["q"]:.5f}|{n["hidden"]:.4f}|{n["fc2"]:.4f}|')
    lines.extend([f'\nIdentityΔq={ident["q"]-base["q"]:+.5f},95%pairedCI{ident["paired_delta_film_bootstrap95"]["q"]}. Identity minus średnia3random Δq={additional["identity_minus_mean_random_q"]:+.5f},CI{additional["film_bootstrap95"]}. Jest większy wpływ na confidence×IoU przy tej samej normie768D, ale fc2normy nie są dopasowane, więc nie dowodzi to wyłącznej semantyki.',
    f'\nIdentityΔAP50={100*(ident["AP50"]-base["AP50"]):+.2f}punktu procentowego,95%CI[{100*ident["AP50_paired_delta_bootstrap95"][0]:+.2f},{100*ident["AP50_paired_delta_bootstrap95"][1]:+.2f}]. Δloss={ident["loss"]-base["loss"]:+.5f},CI{ident["paired_delta_film_bootstrap95"]["loss"]}. Oba CI zawierają0: brak jednoznacznego pogorszenia AP/straty. APto source101punktów jednej klasy, po wszystkich85obrazach; bootstrap po filmach200. Loss/q mają wagi filmowe.',
    f'\nGTcenter w toolROI{det["GT_center_in_roi"]}/85, globalny peaklogit{det["baseline_peak_in_roi"]}/85. W5przypadkach center leży poza maską; ingerencja w shafttokens nie usuwa całej informacji zgromadzonej attention w tokenie centrum. Randomnorm mismatch max{det["random_max_token_norm_mismatch"]:.2g},0zerowychmianowników. Noopmaxerror0; dwie pojedyncze podmiany outputcells sprawdzono prawdziwym hookiem (maxerror0). Pełny decoder uruchomiony ponownie przy każdej podmianie.',
    '\n## Które cząsteczki na obrazie',
    '\nCZASTECZKI_RANKING.json: wszystkie6729tokenów, indeks0/1, row/column, coverage,L2delta i projectionL2; osobny ranking każdego obrazu. CZASTECZKI_WPLYW.json: zmiana metryki po interwencji pojedynczej cząsteczki w6z góry ustalonych testowych przykładach. To różne rankingi; duża delta nie musi obniżać detekcji. Outputcellpatching jest równoważne hookowi dla tej architektury i warstwy, potwierdzone dwiema kontrolami.',
    '\n| source_index | Największa projekcja: token0 / row,col | Największy wpływ: token0 / row,col | Δconfidence×IoU |','|---:|---|---|---:|'])
    for ex in example_descriptions:
        a=ex['largest_projection'];b=ex['largest_metric_change'];lines.append(f'|{ex["source_index"]}|{a["token0"]} / {a["row"]},{a["column"]}|{b["token0"]} / {b["row"]},{b["column"]}|{b["delta_q"]:+.5f}|')
    for ex in example_descriptions:lines.append(f'\n![Reakcja i wpływ source_index{ex["source_index"]}]({ex["figure"]})')
    lines.extend(['\n## Tło i kontrola artefaktów',
    f'\nNa teście średnia ważona projekcji toolROI{bg["tool_projection_mean"]:.5f} vs8farbackgroundtokens{bg["far8_projection_mean"]:.5f}; pełneL2{bg["tool_delta_mean"]:.5f} vs{bg["far8_delta_mean"]:.5f}. Attention może przenosić zmianę na tło; nominalny patch z maską nie jest czystym semantycznym tokenem.',
    f'\nGłówna kontrola24sham z wyłącznie odłożonych filmów: toolROI ΔL2{sh["original_tool_roi_delta_l2"]:.5f},projekcja{sh["original_tool_roi_projection_l2"]:.5f}. Usunięcie rzeczywistego narzędzia na tych samych24źródłach: toolROI ΔL2{sh["removal_tool_delta_mean"]:.5f},projekcja{sh["removal_tool_projection_mean"]:.5f}. ShamROI ΔL2{sh["sham_roi_delta_l2"]:.5f},projekcja{sh["sham_roi_projection_l2"]:.5f}. Średnia zmiana probabilityoriginal−sham{sh["probability_delta"]:+.5f}; zmianycall{sh["presence_call_flips"]}/24. Bada się osobno toolROI i editROI. Pierwsze24mixedsham (18train/3val/3test) pozostają osobną postfitdiagnostyką, nie powiększają testu i nie są scalone przy powtórzonych source_index.',
    '\nSham ma tę samą maskę kształtu/obszaru, przesuniętą na tło, i pozostawia piksele narzędzia; jednak treść i pozycja obszaru różnią się. Bez masek innych instrumentów nie gwarantujemy czystości background. Sham ogranicza najprostsze wyjaśnienie samą edycją, ale nie usuwa confoundingu LaMa i nie dowodzi cystotome versus inne instrumenty.',
    '\n## Co wykazano i czego nie',
    '\nWykazano spójne składowe i małą podprzestrzeń rozdzielającą paired presence, oraz lokalny wpływ wybranych projekcji na confidence×IoU. Nie wykazano jednej wyłącznej tożsamości narzędzia, identyfikacji egzemplarza, całej reprezentacji semantycznej768D, wyraźnego pogorszenia AP/straty, prognozy przyszłych zmian ani treningowego ruchu. LDAklasyfikuje lepiej niżSVD8, ale jego wpływu na detektor w tym protokole nie testowano. Wynik dotyczy jednego checkpointu, źródłowychtrainobrazów i LaMa; wymaga innych instrumentów, niezależnych obrazów i kontroli efektywnej siły po fc2.',
    '\nBłąd XYWH→XYXY został wykryty przez Astrę przed raportem detekcji. Pierwszy bieg hook przerwał się na typie indeksów int16 bez zapisanych metryk; poprawiono torch.long oraz konwersję GT i cały test detekcji policzono od nowa. Podprzestrzeń nie zależy od bbox. Wszystkie raportowane metryki detekcji pochodzą wyłącznie z poprawnego biegu.',
    '\n## Artefakty i odtwarzanie',
    '\nPLAN.md; PODZIAL.json; POMIAR.json; TOKENY_PARY.npz; PODPRZESTRZEN.npz; IDENTYFIKACJA.json; MAKROWEKTORY.json; CZASTECZKI_RANKING.json; CZASTECZKI_WPLYW.json; DETEKCJA.json; DETEKCJA_OBRAZY.json; MAPY_INTERWENCJI.npz; SHAM_DIAGNOSTYKA_heldout.json; SHAM_DIAGNOSTYKA_mixed.json; KATY.json; AUDYT_ASTRA.md. Skrypty pomiar.py,identyfikacja.py,interwencje.py,raport.py uruchamiane kolejno F:/Python/python.exe w tym katalogu. Źródłowych obrazów/wag nie modyfikowano.'])
    (HERE/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HERE/'README.md').write_text('# Cząsteczki i podprzestrzeń narzędzia:400par\n\nZakończona lokalna analiza800obrazów na zamrożonymk1000. [Raport](WYNIKI.md), [plan](PLAN.md), [ranking osi](MAKROWEKTORY.json), [reakcja cząsteczek](CZASTECZKI_RANKING.json), [wpływ cząsteczek](CZASTECZKI_WPLYW.json), [detekcja](DETEKCJA.json), [shamtest](SHAM_DIAGNOSTYKA_heldout.json), [audytAstry](AUDYT_ASTRA.md).\n\nSkrypty uruchomić kolejno w katalogu: pomiar.py, identyfikacja.py, interwencje.py, raport.py z F:/Python/python.exe. Pełne tokeny maskiobu członówfloat32 w TOKENY_PARY.npz; U/probe w PODPRZESTRZEN.npz. Bez treningu/commitów. Ocena dotyczy odpowiedzi na usunięcie, nie ruchu treningowego; syntheticnegative/knownROI i trainmembership ograniczają semantyczne wnioski.\n',encoding='utf-8')
    total=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());free=shutil.disk_usage(HERE).free
    assert total<250e6 and free>1e9
    record=dict(family='maly_vit_tozsamosc_czasteczki_pary_400',class_='badawcza',status='complete_exploratory',created=now,results=fit,presence_definition='original versusLaMa atknownoriginalROI',detection=det,controls=additional,sham=summaries,bytes_total=total,disk_free=free,
        model=measurement,code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},figures=[ex['figure'] for ex in example_descriptions])
    js(HERE/'WYNIKI.json',record)
    print(json.dumps(dict(top_axes=[r['channel1'] for r in axes[:10]],sham=summaries,additional=additional,bytes_total=total,disk_free=free),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
