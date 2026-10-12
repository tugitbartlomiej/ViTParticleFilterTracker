from common import *
from analiza import load_bank
import shutil,time
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def percent(x):return f'{100*x:.2f}%'
def interval(v):return '['+', '.join(f'{100*float(x):.2f}' for x in v)+']%'

def main():
 now=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat();data=read(HERE/'DANE.json');metrics={dt:read(HERE/f'METRYKI_DT{dt}.json') for dt in (1,10)};choices={dt:read(HERE/f'WYBOR_DT{dt}.json') for dt in (1,10)};bank={dt:load_bank(dt) for dt in (1,10)};a=dict(np.load(HERE/'DANE_PROJEKCJE.npz'));fitshared=read(HERE/'RUCH_WSPOLNY_FIT.json')
 names=['oldGLE','localU8','coupledU8','localPCA4','localPCA8','localPLS4','localPLS8','coupledPCA4','coupledPCA8','coupledPLS4','coupledPLS8','best_short']
 lines=['# Wspólny ruch cząsteczek, kontekstu i kierunki poza tożsamością',f'\nZakończono {now}. Analiza eksploracyjna tej samej historii 1001 stanów (0..1000), czyli 1000 aktualizacji; test czasowy był oglądany wcześniej. Bez treningu ViT, pobierania lub refitu na drugiej sondzie.']
 primary=metrics[1];sel=primary['selection'];m=primary['probes']['2']['models'][sel];old=primary['probes']['2']['models']['oldGLE'];config=choices[1]['family_choices'][sel]
 lines.append(f'\nWalidacja wybrała **{sel}, L={config["L"]}, ridge={config["ridge"]}**. Na H1 skill względem braku zmiany to **{percent(m["1"]["skill"])}**, wobec {percent(old["1"]["skill"])} starego GLE; na H10 **{percent(m["10"]["skill"])}**, na H50 **{percent(m["50"]["skill"])}**. To zamrożony wybór walidacyjny, nie najlepsza rodzina wybrana na teście. Samo sprzężenie i dodatkowe kierunki mają osobne ramiona poniżej.')
 lines.extend(['\n## Dane i zamrożone podprzestrzenie',
  f'\nGłówna sonda 2: `{data["probes"]["2"]["frame_id"]}`, źródłowa rola `{data["probes"]["2"]["source_role"]}`, split `{data["probes"]["2"]["split"]}`; {len(data["probes"]["2"]["tool_ids0"])} cząsteczek narzędzia. Transfer 8: `{data["probes"]["8"]["frame_id"]}`, rola `{data["probes"]["8"]["source_role"]}`, split `{data["probes"]["8"]["split"]}`; {len(data["probes"]["8"]["tool_ids0"])} cząsteczek. Transfer ma własne maski/coverage i anchor500, lecz dokładnie te same U,V oraz operatory i skale. Nie jest nowym biegiem ani dowodem generalizacji populacyjnej.',
  '\nU500 (768×8) pochodzi wyłącznie z wcześniejszych 240 par train na k500. Cel każdej cząsteczki to pełny q=Uᵀz w 8D. Dodatkowe r=Vᵀz służą tylko jako pomocnicze współrzędne; nie zastępują celu i nie oznaczają semantycznej tożsamości. V ma 4 albo 8 kierunków poza U. Wszystkie q/r oraz średnie tła w przyszłości są prognozowane autonomicznie.',
  '\nPCA: ważona covariance wycentrowanych residualnych przyrostów (I−UUᵀ)Δz, targets501..700. PLS/SVD: crosscov residualnego Δz_t (t500..699) z Δq_(t+1), zakończone targets501..700. Oba centrowania i wagi coverage×1/200 tylko fit głównej sondy. PLS jest tu liniowym SVD crosscov, bez autoenkodera. Rangi 4/8 to prefiksy basis8. V ponownie ortogonalizowano względem U i metodą QR; obie bazy zamrożono przed walidacją i nie refitowano przy operatorach do799.',
  f'\nMaksymalne |UᵀV|: PCA {data["PCA"]["U_V_max"]:.3g}, PLS {data["PLS"]["U_V_max"]:.3g}. Surowe 768D wszystkich tokenów narzędzia i pełne średnie tła odczytano przez mmap, wyłącznie do RAM. Dysk przechowuje małe projekcje, nie kopię 22 GB. Zgodność nowych q i dokładnych starych prognoz ma maks. błąd 0.',
  '\n## Równanie i identyfikacja',
  '\n```text\ns_i(t)=[Uᵀz_i(t),Vᵀz_i(t)] − [Uᵀz_i(500),Vᵀz_i(500)]\ng_fg(t)=sum_i w_i s_i(t)\ng_bg(t)=[U,V]ᵀ mean_background z(t) − jego wartość w500\nΔs_i(t+dt)=b+A s_i(t)+C_fg g_fg(t)+C_bg g_bg(t)\n              +sum_l [K_l Δs_i(t−l dt)+F_l Δg_fg(t−l dt)+G_l Δg_bg(t−l dt)]\n              +epsilon_i(t+dt)\nΔg_bg(t+dt)=d+E_bg g_bg(t)+E_fg g_fg(t)\n              +sum_l [M_l Δg_bg(t−l dt)+N_l Δg_fg(t−l dt)]+eta(t+dt)\n```',
  '\nWariant lokalny usuwa C/F/G i operator tła. CoupledU8 ma V puste; pozostałe ramiona dodają PCA/PLS. g_fg ZAWSZE wynika z aktualnych prognoz cząsteczek, nie ma osobnego fitu średniej. Aktualizacje foreground/bg są synchroniczne: oba przyrosty z tego samego snapshotu, dopiero potem podmiana stanów i pamięci. Wszystkie macierze foreground są wspólne dla cząsteczek. To niskorzędowe sprzężenie ze wspólnym ruchem, nie identyfikacja indywidualnych sił między tokenami.',
  '\nDokładna dekompozycja deterministycznej części tego modelu: Δg_fg=b+(A+C_fg)g_fg+C_bg g_bg+średnia pamięci; Δ(s_i−g_fg)=A(s_i−g_fg)+pamięć własnych odchyleń. Wspólny człon kontekstu znosi się w różnicach między cząsteczkami. Model nie różnicuje reakcji tokenów na ten sam kontekst; A/C_fg nie są osobno udowodnionymi mechanizmami fizycznymi.',
  '\nForeground fit: każda zakończona aktualizacja ma tę samą wagę, wewnątrz niej coverage-weighted tokeny. Background fit: jedna średnia na aktualizację, równe wagi czasu. Wszystkie poziomy centrowane tokenowym anchor500. Każda cecha/laga ma osobny historyczny SD. Ridge wspólnie estymuje dryf, sprzężenie i pamięć; gamma=lambda×trace(Gram)/liczba_cech, intercept bez kary. To inne skalowanie i siatka niż dawny GLE; dlatego nowy localU8 jest obowiązkową kontrolą, a oldGLE pozostaje dokładną historyczną bazą.',
  '\nW zapisie wierszowym F@B, B w pliku jest już fizyczne: B=diag(1/s_x) B_std diag(s_y), b=mean_y−mean_x B. Kolumnowe A/C/K są transponowanymi blokami B zgodnie z layout_fg; operator tła analogicznie layout_bg. Poziomy i przyrosty mają jednostki projekcji aktywacji; B jest bezwymiarowe dla kroku dt, b/reszta są przyrostami na ten krok. W forecast odwrócono wszystkie skale. epsilon_q w metadanych jest progiem znaku, nie realizacją reszty równania.',
  '\nNie zapisano faktycznych minibatch-ID/augmentacji/gradientów każdej aktualizacji, więc brak jawnego historycznego sterowania. Nie podano przyszłych paczek, obrazów, stanów, r, średnich ani reszt. Źródłowy eval obejmuje wagi i biegnące statystyki BatchNorm. Nie zakładamy Brownowskiego szumu, FDT dla AdamW ani dokładnego Mori–Zwanziga/globalnej liniowości. Dodatniego wyniku nie utożsamiamy z semantyczną swoistością.',
  '\n## Wybór przed testem',
  '\n10 nowych rodzin, ridge .001/.01/.1/1/10, L0/2/4/8 dla dt1 oraz L0/1/2/4/8 dla dt10. Każda rodzina wybiera swoją konfigurację wyłącznie na walidacji; również dokładny oldGLE może być głównym wyborem. Cel wyboru to średni skill fizycznego 8D q, równoważnie po H1/10/50 (dt1) albo H10/50 (dt10). Osobny best_short maksymalizuje H1/H10. Bez wyboru rodziny na teście.',
  '\ndt1: fit targets501..700; validation origins700..749, cele<=799; operator refit do799, V bez refitu. Test origins800..950 dla wszystkich H1/2/5/10/20/50, 151 originów. dt10: targets510..700, origins700..740 co10, refit do790; test800..950 co10, 16 originów. Porównania dt1/dt10 używają zapisanej sekcji common_co10, tych samych 16 originów. Nie dodawano H100 ani nowych siatek po teście.',
  '\nPrefix oraz zatrucie przyszłego q/r/bg przez NaN dają identyczny rollout przy origin800/950 dla każdej nowej rodziny i obu sond; globalne średnie powstają wewnątrz rollout z prefixu. U/V są już zamrożone, a projekcja/maskowanie w przygotuj.py jest pointwise w czasie. SHA modeli i danych są sprawdzane przed testem. Pełne wyniki wyboru, również duże/niestabilne wartości, pozostają w WYBOR_DT*.json.'])
 lines.extend(['\n## Sonda główna: zamrożone rodziny', '\n| Rodzina | L/ridge dt1 | H1 skill | H10 skill | H50 skill | H1 znak wszystkich |','|---|---:|---:|---:|---:|---:|'])
 for name in names:
  if name not in primary['probes']['2']['models']:continue
  r=primary['probes']['2']['models'][name];conf=bank[1].get(name);cfg=f'{conf["L"]}/{conf["ridge"]}' if conf else 'stary zamrożony'
  lines.append(f'|{name}|{cfg}|{percent(r["1"]["skill"])}|{percent(r["10"]["skill"])}|{percent(r["50"]["skill"])}|{percent(r["1"]["sign_accuracy"])}|')
 lines.extend(['\n## Wybrany model: błąd, kierunek i kontrast do starego GLE', '\n| H | MSE | Skill vs brak zmiany | Znak wszystkich | Znak po odsiewie celu | Cosine zmiany | Δskill vs oldGLE, CI95 [pp] |','|---:|---:|---:|---:|---:|---:|---|'])
 for H in (1,2,5,10,20,50):
  r=m[str(H)];c=primary['probes']['2']['contrasts_vs_oldGLE'][sel][str(H)]
  lines.append(f'|{H}|{r["MSE"]:.6f}|{percent(r["skill"])}|{percent(r["sign_accuracy"])}|{percent(r["sign_filtered"])}|{r["cosine"]:.4f}|{100*c["point"]["skill_difference_vs_old"]:.2f} pp; {interval(c["CI95"]["skill_difference_vs_old"]).replace("%"," pp")}|')
 mean_new=np.mean([m[str(H)]['skill'] for H in (1,10,50)]);mean_old=np.mean([old[str(H)]['skill'] for H in (1,10,50)])
 h1contrast=primary['probes']['2']['contrasts_vs_oldGLE'][sel]['1'];ci=h1contrast['CI95']['MSE_reduction_vs_old']
 lines.append(f'\nŚredni testowy skill równych H1/10/50: nowy {percent(mean_new)}, stary {percent(mean_old)}, różnica {100*(mean_new-mean_old):.3f} punktu procentowego. Walidacyjna przewaga tego kryterium praktycznie się nie przeniosła. Na H1 MSE jest większy o {100*(m["1"]["MSE"]/old["1"]["MSE"]-1):.2f}% od starego, z opisowym CI pogorszenia {interval([-ci[1],-ci[0]])}.')
 lines.append('\nWszystkie metody ocenione według fizycznego q, z identycznymi coverageweights, originami, celem i progami dawnego fit do799. Znak to poprawność składowych, nie całych wektorów; cosine dotyczy wektora zmiany. Persistence ma zerowy wektor zmiany i nieokreśloną cosine. Odsiew abs(rzeczywistej przyszłej zmiany)>próg jest warunkową metryką dopiero po poznaniu celu, nie bramką pewności online.')
 lines.extend(['\n## Transfer na sondę 8 bez refitu', '\n| Rodzina | H1 skill | H10 skill | H50 skill | H1 znak |','|---|---:|---:|---:|---:|'])
 for name in names:
  if name not in primary['probes']['8']['models']:continue
  r=primary['probes']['8']['models'][name];lines.append(f'|{name}|{percent(r["1"]["skill"])}|{percent(r["10"]["skill"])}|{percent(r["50"]["skill"])}|{percent(r["1"]["sign_accuracy"])}|')
 lines.extend(['\n## Skala co10: wspólne originy', '\n| H | dt1 selected, co10 | oldGLE dt1, co10 | dt10 selected | oldGLE dt10 |','|---:|---:|---:|---:|---:|'])
 selected10=metrics[10]['selection'];m10=metrics[10]['probes']['2']['models'][selected10]
 for H in (10,20,50):lines.append(f'|{H}|{percent(primary["probes"]["2"]["common_co10"][sel][str(H)]["skill"])}|{percent(primary["probes"]["2"]["common_co10"]["oldGLE"][str(H)]["skill"])}|{percent(m10[str(H)]["skill"])}|{percent(metrics[10]["probes"]["2"]["models"]["oldGLE"][str(H)]["skill"])}|')
 c10=choices[10]['family_choices'][selected10];lines.append(f'\ndt10 walidacja wybrała {selected10}, L={c10["L"]}, ridge={c10["ridge"]}. L0 oznacza brak jawnych lagów przyrostów w rozszerzonym stanie. Eliminacja pomocniczych r/bg może nadal dawać efekty pamięci w samym q. Wszystkie rodziny dt10, transfer i kontrasty są w METRYKI_DT10.json.')
 lines.extend(['\n## Ranga, stabilność i ograniczenia identyfikacji','\n| dt / model / operator | Targets refit | Cecha count | Gram rank | Condition z ridge | Błąd raw/skal |','|---|---:|---:|---:|---:|---:|'])
 spectrum_notes=[]
 for dt in (1,10):
  for name in (metrics[dt]['selection'],'coupledU8'):
   mod=bank[dt].get(name)
   if mod is None:continue
   for which in ('fg','bg'):
    op=mod[which]
    if op is None:continue
    lines.append(f'|{dt} / {name} / {which}|{len(mod["fit_targets"])}|{op["features"]}|{op["gram_rank"]}|{op["condition"]:.3g}|{op["native_max_error"]:.3g}|')
   spectrum_notes.append(f'\n{name}, dt{dt}: spectral radius odchyleń = {mod["spectrum"]["deviation_radius"]:.6f}; średniaFG/bg = {mod["spectrum"]["mean_background_radius"]}.')
 lines.extend(spectrum_notes)
 lines.append('\nAblation: przy dt1 coupledU8 poprawia H50 względem nowego localU8 (10.58% vs −5.03%), kosztem H1 (49.81% vs 60.24%). LocalPLS8 vs coupledPLS8 na H50: 0.18% vs 14.19%; oba są skonfigurowane na walidacji, ale nie wybieramy nowego zwycięzcy po tych wynikach. Przy dt10 przewaga coupledU8 na sondzie2 nie przenosi się na8: H50 −42.79% vs −0.29% localU8.')
 lines.append('\nOperator tła ma tylko200/20 targetów pierwotnego fit i299/29 po reficie, niezależnie od liczby tokenów. Przy dużym D/L liczba cech przekracza liczbę czasowych obserwacji; ridge daje predyktor, nie jednoznaczną identyfikację dużej macierzy. Powtarzane globalne cechy w foreground nie tworzą nowych niezależnych próbek. Same parametry nie dowodzą obserwowalności/sterowalności rzeczywistego procesu.')
 lines.append(f'\nWyłącznie fit501..700: udział energii wspólnego przyrostu meanfg = {percent(fitshared["common_energy_fraction"])}, względnej = {percent(fitshared["relative_energy_fraction"])}; tożsamość dekompozycji ma relative error {fitshared["identity_relative_error"]:.3g}. Średni cosine przyrostów meanfg/bg = {fitshared["mean_fg_bg_cosine"]:.4f}. To podział energii, nie trafność prognozy ani semantyczna odpowiedzialność.')
 lines.append('\nCI to sparowany circular moving block bootstrap300, bloki co najmniej50 aktualizacji, wspólne originy/tokens/models. Przy151/16originach około3 efektywne bloki: niepewność opisowa, jeden bieg i dwa obrazy, nie tokeny jako replikacje. Powtórnie analizowany test pozostaje eksploracyjny. Kolorowość reszty i gaussowskość są odrębne; wcześniejszy audyt szumu nie jest dowodem białości ani odzyskania ukrytego sygnału przez nowe sprzężenie.')
 lines.extend(['\n## Nowe reszty wybranych modeli', '\nReszty one-step obliczono po dopasowaniu z ciągłej historii target801..1000 (dt1, 200 próbek) oraz810..1000 (dt10, 20 próbek). Wejście każdej reszty ma rzeczywistą historię do originu; są to diagnostyki niedomknięcia, odrębne od autonomicznych błędów H10/H50 i niewykorzystane do strojenia.', '\n| dt / sonda / model | Szeregi | Ljung–Box raw p<.05 | Jarque–Bera raw p<.05 | Udział wspólnej energii reszty |','|---|---:|---:|---:|---:|'])
 for dt in (1,10):
  ns=read(HERE/f'SZUM_DT{dt}.json')
  for key,r in ns['records'].items():lines.append(f'|{dt} / {key}|{r["series_count"]}|{r["LB_rejections_raw"]}|{r["JB_rejections_raw"]}|{percent(r["shared_energy_fraction"])}|')
 lines.append('\nACF/Ljung–Box są liczone osobno w czasie dla każdej cząsteczki/współrzędnej i średniej, bez spłaszczania tokenów. Normalność badana oddzielnie. Raw p bez korekty wielokrotnych porównań i liczby dopasowanych parametrów; dt10 ma bardzo mało próbek. Nowy dt1 zachowuje silne korelacje reszty (256/256 odrzuceń Ljung–Box), więc nie wykazano wybielenia szumu ani wydobycia jego całego przewidywalnego składnika. Pełne ACF, korelacje między tokenami i wariancje połów zapisano w SZUM_DT*.json. Brak odrzucenia normalności nie dowodzi gaussowskości ani białości.')
 # figures
 hs=[1,2,5,10,20,50];fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 for name in ['oldGLE','localU8','coupledU8',sel,'coupledPLS8']:
  r=primary['probes']['2']['models'][name];axs[0].plot(hs,[100*r[str(H)]['skill'] for H in hs],marker='o',label=name);axs[1].plot(hs,[100*r[str(H)]['sign_accuracy'] for H in hs],marker='o',label=name)
 for ax in axs:ax.set_xscale('log');ax.set_xlabel('H [aktualizacje]');ax.grid(alpha=.2);ax.legend(fontsize=7)
 axs[0].axhline(0,color='gray');axs[0].set_ylabel('Skill [%]');axs[1].axhline(80,color='gray',ls=':');axs[1].set_ylabel('Znak wszystkich składowych [%]');fig.tight_layout();fig.savefig(HERE/'POROWNANIE_HORYZONTOW.png',dpi=135);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5))
 for probe in ('2','8'):
  for name in [sel,'oldGLE']:
   r=primary['probes'][probe]['models'][name];ax.plot(hs,[100*r[str(H)]['skill'] for H in hs],marker='o',ls='-' if probe=='2' else '--',label=f'sonda{probe}, {name}')
 ax.set_xscale('log');ax.axhline(0,color='gray');ax.set_xlabel('H');ax.set_ylabel('Skill [%]');ax.legend();fig.tight_layout();fig.savefig(HERE/'TRANSFER_SONDA8.png',dpi=135);plt.close(fig)
 q=a['s_2_PCA'][:,:,:8];w=a['w_2'];fg=np.einsum('tnd,n->td',q-q[500],w);bg=a['bg_2_PCA'][:,:8]-a['bg_2_PCA'][500,:8];fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True)
 for j,ax in enumerate(axs):ax.plot(fg[:,j],label=f'meanfg q{j+1}');ax.plot(bg[:,j],label=f'background q{j+1}');ax.axvline(500,color='gray',ls='--');ax.axvline(800,color='red',ls=':');ax.legend();ax.set_ylabel('Aktywacja względem anchor500')
 axs[-1].set_xlabel('Aktualizacja');fig.suptitle('Wspólny ruch narzędzia i kontekstu; obserwacja, nie wejście przyszłe');fig.tight_layout();fig.savefig(HERE/'RUCH_FG_BG.png',dpi=135);plt.close(fig)
 preds=np.load(HERE/'PROGNOZY_DT1.npz');origins=np.arange(800,951);ids=a['ids_2'];boundary=int(np.argmin(data['probes']['2']['coverage']));interior=int(np.argmax(data['probes']['2']['coverage']));fig,axs=plt.subplots(3,2,figsize=(12,8),sharex=True)
 for row,H in enumerate((1,10,50)):
  for col,k in enumerate((boundary,interior)):
   ax=axs[row,col];truth=q[origins+H,k,0]-q[origins,k,0];ax.plot(origins,truth,color='black',label='rzeczywista Δq1')
   for name in [sel,'oldGLE']:ax.plot(origins,preds[f'probe2_{name}_H{H}'][:,k,0]-q[origins,k,0],lw=.8,label=name)
   ax.set_title(f'H{H}; token{ids[k]+1} [od0:{ids[k]}]');ax.axhline(0,color='gray',ls=':')
 axs[0,0].legend(fontsize=7);axs[-1,0].set_xlabel('Origin');axs[-1,1].set_xlabel('Origin');fig.tight_layout();fig.savefig(HERE/'PROGNOZY_INDYWIDUALNE.png',dpi=135);plt.close(fig)
 lines.extend(['\n![Porównanie rodzin](POROWNANIE_HORYZONTOW.png)','\n![Transfer bez refitu](TRANSFER_SONDA8.png)','\n![Ruch średnich](RUCH_FG_BG.png)','\n![Prognozy cząsteczek](PROGNOZY_INDYWIDUALNE.png)','\n## Artefakty i odtwarzanie','\nF:/Python/python.exe przygotuj.py; analiza.py --stage validation; audyt Astry; analiza.py --stage test; reszty.py; raport.py. Przygotuj wymaga PLAN.md. Nie uruchamia żadnego forward/treningu ViT. DANE_PROJEKCJE.npz zawiera U,V i małe projekcje obu sond; MACIERZE_DT*.npz oraz MODELE_DT*.json to fizyczne operatory, skale i jawny layout. WYBOR_DT*.json zachowuje całą siatkę; METRYKI_DT*.json wszystkie rodziny, per-token/per-coordinate błędy, transfer, common_co10 i kontrasty z CI. PROGNOZY_DT*.npz zapisuje tylko wybrane prognozy i bazę. AUDYT_PREFIX_DT*, ZGODNOSC_STARY_GLE.json, RUCH_WSPOLNY_FIT.json i AUDYT_ASTRA.md dokumentują weryfikację. PLAN/PODSUMOWANIE należą do roota.'])
 text='\n'.join(lines)+'\n'
 for old,new in {'targets501': 'targets 501', 't500': 't 500', 'do799': 'do 799', 'basis8': 'bazy 8', 'w500': 'w 500', 'anchor500': 'anchor 500', 'origins700': 'origins 700', 'origins800': 'origins 800', 'targets510': 'targets 510', 'do790': 'do 790', 'co10': 'co 10', 'common_co 10': 'common_co10', 'origin800': 'origin 800', 'coverageweights': 'wagi coverage', 'sondzie2': 'sondzie 2', 'na8:': 'na sondę 8:', 'tylko200': 'tylko 200', 'i299': 'i 299', 'fit501': 'fit 501', 'bootstrap300': 'bootstrap 300', 'najmniej50': 'najmniej 50', 'Przy151/16originach': 'Przy 151/16 originach', 'około3': 'około 3', 'target801': 'target 801', 'oraz810': 'oraz 810'}.items():text=text.replace(old,new)
 (HERE/'WYNIKI.md').write_text(text,encoding='utf-8');(HERE/'README.md').write_text('# Wspólny ruch i pomocnicze kierunki\n\nZakończony eksploracyjny pilot: [wyniki](WYNIKI.md), [plan](PLAN.md), [audyt](AUDYT_ASTRA.md), [podsumowanie](PODSUMOWANIE.md). Bez nowych danych/treningu.\n\nOdtwarzanie: `F:/Python/python.exe przygotuj.py`, potem `analiza.py --stage validation`, audyt, `analiza.py --stage test`, `reszty.py`, `raport.py`. U500 i V fit-only są zamrożone; transfer8 bez refitu. Wszystkie przyszłe q/r/średnie są przewidywane autonomicznie. Dokładny dawny GLE i nowy localU8 są osobnymi kontrolami.\n',encoding='utf-8')
 size=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());free=shutil.disk_usage(HERE).free;assert size<150e6 and free>1e9
 js(HERE/'WYNIKI.json',dict(status='complete_exploratory',date=now,selected={str(dt):metrics[dt]['selection'] for dt in (1,10)},data=data,metrics_files={str(dt):sha(HERE/f'METRYKI_DT{dt}.json') for dt in (1,10)},bytes_total=size,free_disk=free,fit_shared_energy=fitshared,code_sha={p.name:sha(p) for p in HERE.glob('*.py')}))
 print('REPORT',size,'bytes',free,'free',flush=True)
if __name__=='__main__':main()
