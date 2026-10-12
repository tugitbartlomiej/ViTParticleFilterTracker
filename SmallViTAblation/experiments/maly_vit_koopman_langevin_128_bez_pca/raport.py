"""Raportuje całe porównanie, w tym przegrane z kontrolami; nie wybiera modeli po teście."""
from dane import HERE, SOURCE, read, write_json, sha, np
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

NAMES = {'langevin':'Langevin z pamięcią', 'koopman_przyrost':'Koopman przyrostowy',
         'koopman_stan':'Koopman bezpośredni', 'velocity':'Stała prędkość', 'persistence':'Brak zmiany'}
COLORS = {'langevin':'#1976b8','koopman_przyrost':'#289447','koopman_stan':'#e58217','velocity':'#8e62b5'}
data = read(HERE/'METRYKI.json')
freeze = read(HERE/'WYBOR_PRZED_TESTEM.json')
models = read(HERE/'MODELE.json')
audit = read(HERE/'AUDYT.json')
records = data['records']

def plot(probe, field, filename):
    fig, axes = plt.subplots(1,2,figsize=(13,6))
    for ax, dt in zip(axes,(1,10)):
        rec=records[str(dt)][str(probe)]
        hs=rec['horizons']
        for name,color in COLORS.items():
            vals=[100*rec['models'][name][str(H)][field] for H in hs]
            ax.plot(hs,vals,'o-',color=color,label=NAMES[name],linewidth=2)
        ax.set_xscale('log');ax.set_xticks(hs,[str(H) for H in hs])
        ax.minorticks_off();ax.grid(alpha=.2)
        ax.set_title(f'Zapis co {dt} aktualizacji')
        ax.set_xlabel('Horyzont [aktualizacje]')
        if field=='sign_accuracy':
            ax.set_ylim(0,100);ax.set_ylabel('Trafność znaku zmian [%]')
        else:
            ax.set_yscale('symlog',linthresh=10)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:g}'))
            ax.axhline(0,color='gray',linestyle='--',linewidth=1)
            ax.set_ylabel('Redukcja MSE względem braku zmiany [%]')
    title='Wszystkie 128 oryginalnych kanałów, bez PCA'
    title += f' | obraz {probe}' + (' (transfer bez dopasowania)' if probe==8 else '')
    fig.suptitle(title,fontsize=15,y=.98)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.subplots_adjust(top=.75,bottom=.13,left=.075,right=.98,wspace=.25)
    fig.savefig(HERE/filename,dpi=160)
    fig.savefig(HERE/filename.replace('.png','.svg'))
    plt.close(fig)

for probe in (2,8):
    plot(probe,'sign_accuracy',f'ZNAK_OBRAZ_{probe}.png')
    plot(probe,'skill',f'BLAD_OBRAZ_{probe}.png')

with (HERE/'POROWNANIE.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['dt','probe','model','H','MSE','skill_percent','sign_percent','cosine'])
    writer.writeheader()
    for dt,probes in records.items():
        for probe,rec in probes.items():
            for name,scores in rec['models'].items():
                for H,v in scores.items():
                    writer.writerow(dict(dt=dt,probe=probe,model=name,H=H,MSE=v['MSE'],skill_percent=100*v['skill'],sign_percent=100*v['sign_accuracy'],cosine=v['cosine']))

def pct(x): return f'{100*x:.2f}%'
def score(dt,probe,name,H,field): return records[str(dt)][str(probe)]['models'][name][str(H)][field]

lines=['# Koopman i Langevin: te same 128 oryginalnych kanałów',
       '',f'Obliczenia zakończono {data["completed"]}. Bez PCA, rotacji i ponownego wyboru kanałów.',
       '', 'Policzono samodzielne prognozy dwóch liniowych przybliżeń Koopmana i odtworzono prognozy zamrożonego Langevina. Każda cząsteczka ma dokładnie te same 128 wybranych współrzędnych jako stan i cel. Źródłowy wynik 92.83 procent dotyczy rozpoznania obecności narzędzia w parach oryginał/LaMa na k500, a nie prognozy ruchu. Pełny latent ma nadal 768 współrzędnych.',
       '', '## Równania i znaczenie porównania',
       '', 'Stan x_i(t) = z_i,J(t) − z_i,J(500), J zapisano w WYBOR_PRZED_TESTEM.json. Wagi cząsteczek odpowiadają pokryciu maską narzędzia.',
       '', '```math',r'\widehat{x}_{i,t+\Delta t}=Kx_{i,t}+b \qquad\text{(Koopman)}',
       r'\widehat{x}_{i,t+\Delta t}=x_{i,t}+b+Ax_{i,t}+K_0\Delta x_{i,t}+K_1\Delta x_{i,t-\Delta t} \qquad\text{(Langevin)}','```',
       '', 'K, A oraz obie macierze pamięci są pełnymi macierzami 128×128. Współczynniki są wspólne dla cząsteczek. Stała w słowniku Koopmana obsługuje wyraz wolny; macierz rozszerzona ma 129×129, ale nie dodaje nowego kanału obrazu. To przybliżenie operatora Koopmana na funkcjach (1,x), bez dowodu dokładnego domknięcia dynamiki w tej przestrzeni.',
       '', 'Koopman przyrostowy ma K = I + A. Jest to wcześniejszy wariant bez jawnej pamięci L0, z regularyzacją ku brakowi ruchu. Niezależny fit i osobny rollout odtwarzają jego macierze i wyniki. Zmiana zapisu na K nie stanowi nowej metody ani poprawy.',
       '', 'Koopman bezpośredni dopasowuje następny stan z regularyzacją ku K = 0. Ma tę samą klasę funkcji affine co wariant przyrostowy, ale inną karę. Wybrano ridge '+str(models['1']['koopman_stan']['ridge'])+' dla dt1 oraz '+str(models['10']['koopman_stan']['ridge'])+' dla dt10, wyłącznie na walidacji ze stałej siatki sześciu wartości.',
       '', 'Langevin używa dwóch przyrostów historii i wcześniej wybranej regularyzacji 1 dla dt1 oraz 10 dla dt10. Zachowano jego macierze, ale przeliczono prognozy osobną implementacją. W prognozie punktowej nie podajemy przyszłych realizacji szumu. Poprzednia diagnostyka reszt nie potwierdziła białości, więc nie traktujemy modelu jako dokładnie domkniętego równania stochastycznego.',
       '', '## Wyniki na obrazie 2',
       '', '| Model, zapis co jedną aktualizację | Znak H1 | Redukcja MSE H1 | Redukcja MSE H10 | Redukcja MSE H50 |',
       '|---|---:|---:|---:|---:|']
for name in NAMES:
    lines.append('|'+NAMES[name]+'|'+pct(score(1,2,name,1,'sign_accuracy'))+'|'+'|'.join(pct(score(1,2,name,H,'skill')) for H in (1,10,50))+'|')
lines += ['', 'Redukcja MSE = 100 × (1 − MSE_model / MSE_brak_zmiany). Wynik ujemny oznacza gorszą prognozę niż brak ruchu, a nie procent błędnie rozpoznanych obrazów. Trafność znaku jest średnią po wszystkich współrzędnych z wagami maski. Nie odsiewamy małych zmian na tych wykresach. Brak zmiany nie przewiduje wzrostu ani spadku, stąd zerowa lub bliska zeru trafność znaku w tej definicji.',
          '', '| Model, zapis co dziesięć aktualizacji | Znak H10 | Redukcja MSE H10 | Redukcja MSE H20 | Redukcja MSE H50 |',
          '|---|---:|---:|---:|---:|']
for name in NAMES:
    lines.append('|'+NAMES[name]+'|'+pct(score(10,2,name,10,'sign_accuracy'))+'|'+'|'.join(pct(score(10,2,name,H,'skill')) for H in (10,20,50))+'|')
lines += ['', '## Drugi obraz: transfer bez refitu', '', '| dt | Model | Redukcja MSE H1 | Redukcja MSE H10 | Redukcja MSE H50 |','|---:|---|---:|---:|---:|']
for dt in (1,10):
    for name in ('langevin','koopman_przyrost','koopman_stan','velocity'):
        vals=[pct(score(dt,8,name,H,'skill')) if H>=dt else 'n/d' for H in (1,10,50)]
        lines.append(f'|{dt}|{NAMES[name]}|'+'|'.join(vals)+'|')
lines += ['', '## Ocena poza dopasowaniem i ograniczenia',
          '', 'Fit: aktualizacje do 700; walidacja do 799. Końcowy fit operatorów sięga 799 dla dt1 albo 790 dla dt10. Test: starty 800–950, horyzont do 50 aktualizacji. Obraz 2 ma 31 cząsteczek narzędzia, obraz 8 ma 7. Nie dopasowywano operatorów na drugim obrazie. Wszystkie dane pochodzą z jednego biegu sieci. Cząsteczki i nakładające się okna nie są niezależnymi powtórzeniami.',
          '', 'Test był już oglądany w poprzednich analizach. W tym rozszerzeniu przestrzegano zamrożonej siatki i wyboru na walidacji, ale wynik pozostaje eksploracyjny. Nie zmieniano kanałów, siatki ani wyboru po nowych wynikach. MODELE.json podaje promienie spektralne Koopmana; nie narzucono stabilności po fakcie. Wyniki co jedną i co dziesięć aktualizacji mają różne gęstości startów; METRYKI.json zawiera również ocenę dt1 na wspólnych startach co10.',
          '', 'Modele mają ten sam fizyczny cel, skalowanie wyłącznie na fit i ten sam dostęp do zakończonej historii. Różni się wykorzystana długość pamięci. Langevin ma 49280 parametrów, każdy Koopman 16512; porównanie jest poza dopasowaniem, nie na błędzie treningowym. Nie używamy przyszłych klatek, przyszłych minibatchy ani gradientów. Nie ma telemetrii sterowania tego biegu, więc w tych modelach nie dodano jawnego członu obrazu/batcha.',
          '', '## Kontrole i artefakty',
          '', f'Wszystkie {len(audit["checks"])} kontrole numeryczne zapisane w AUDYT.json przeszły. Obejmują odtworzenie źródłowego L0 przez osobne dopasowanie Koopmana przyrostowego, zgodność nowej implementacji prognozy Langevina z zapisanymi prognozami, testy prefix/poison przyszłych stanów i cały wymiar128. Dodatkowo kod sprawdza zgodność wyników MSE i znaków z zamrożonymi ocenami oraz hashe źródeł. To kontrola numeryczna kodu, nie nowy niezależny audyt Astry.',
          '', 'MACIERZE.npz: pełne K i b obu operatorów Koopmana oraz A, K0, K1 i b Langevina. WYBOR_PRZED_TESTEM.json: kanały J, siatka, wybór i hashe zapisane przed oceną. PROGNOZY.npz: pełne prognozy128D. METRYKI.json: wszystkie metryki i opisowe przedziały blokowe. POROWNANIE.csv: płaska tabela do dalszego porównania.',
          '', '![Trafność znaku](ZNAK_OBRAZ_2.png)', '', '![Błąd](BLAD_OBRAZ_2.png)',
          '', '![Transfer znak](ZNAK_OBRAZ_8.png)', '', '![Transfer błąd](BLAD_OBRAZ_8.png)',
          '', '## Odtworzenie', '', '```powershell', "& 'F:\\Python\\python.exe' porownanie.py", "& 'F:\\Python\\python.exe' raport.py", '```',
          '', 'Skrypt porównania blokuje nadpisanie istniejących metryk, aby nie powtarzać strojenia na znanym teście. Raport można regenerować bez fitu. Źródła i wcześniejsze eksperymenty pozostają niezmienione.']
(HERE/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(HERE/'README.md').write_text('# Koopman i Langevin w oryginalnych 128 kanałach\n\nGotowe. Wyniki w WYNIKI.md, równania w MACIERZE.npz, wykresy ZNAK/BLAD_OBRAZ_2/8. Bez PCA.\n\nUruchomienie: F:/Python/python.exe porownanie.py, następnie raport.py. Porównanie blokuje nadpisanie istniejącego testu. Dane ładuje dane.py z zamrożonego eksperymentu źródłowego.\n',encoding='utf-8')
write_json(HERE/'WYNIKI.json',dict(rodzina=HERE.name,klasa='badawcza',status='complete',
    pytanie='Koopman i Langevin na tych samych128 oryginalnych kanałach bez PCA',
    date=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),source=SOURCE,channels=128,PCA=False,
    exploratory=True,source_hashes=freeze['sources'],checks_passed=audit['all_passed'],
    outputs={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='WYNIKI.json'}))
print('Report and four figures complete',flush=True)
