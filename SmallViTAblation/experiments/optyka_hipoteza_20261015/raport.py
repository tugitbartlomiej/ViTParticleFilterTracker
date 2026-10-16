"""Build a reproducible Polish report from measured outputs; fail on failed QA."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/optyka_hipoteza_20261015';CODE=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def link(p):return p.as_posix()
def fmt(x):return f'{x:.2f}'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ext=read('ekstrakcja.json');pred=read('prognozy.json');ph=read('faza.json');raw=read('faza_poziomy.json');cx=read('zapis_zespolony.json')
    desc=read('opis_superpozycji.json');d=np.load(OUT/'pomiar.npz');p=np.load(OUT/'prognozy.npz')
    assert len(d['steps'])==806 and len(set(ext['image_hashes']))==3
    assert all(len(np.unique(ids))==24 for ids in d['ids'])
    assert all(np.isfinite(d[k]).all() for k in ['features','x_pre','x_post','u','y','raw'])
    assert not (p['train']&p['val']).any() and not (p['train']&p['test']).any() and not (p['val']&p['test']).any()
    assert not p['train'][:,2].any() and not p['val'][:,2].any()
    assert max(c['residual_relative'] for c in ext['checks'])<1e-5
    assert max(c.get('head_jvp_fd_relative',0) for c in cx['checks'])<1e-6
    assert max(r['relative_U_error_median'] for r in cx['rows'] if r['rank']==192)<1e-10
    for name,rows in [('prognozy_tabela.csv',pred['evaluation']),('faza_tabela.csv',ph['evaluation']),('zapis_zespolony_tabela.csv',cx['rows'])]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Show precisely which images/tokens entered the probe.
    fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    colors=['#0072B2','#D55E00','#009E73']
    for im,n in enumerate(ext['names']):
        h=np.load(ROOT/'app/cache_uczenia/lasery'/n/'dane.npz');ax[im].imshow(h['rgb'])
        g=d['gt'][im];ax[im].add_patch(Rectangle(g[:2],g[2]-g[0],g[3]-g[1],fill=False,color='yellow',lw=1.5))
        for role in range(3):
            ids=d['ids'][im,8*role:8*(role+1)]
            ax[im].scatter((ids%30+.5)*16,(ids//30+.5)*16,s=14,c=colors[role],marker='x')
        ax[im].set_title(f'Obraz {im+1}; token środkowy {d["ids"][im,0]}'+(' — poza dopasowaniem' if im==2 else ''))
        ax[im].axis('off')
    fig.suptitle('Trzy istniejące historie; żółta ramka = anotacja; punkty = stały wybór geometryczny')
    fig.savefig(OUT/'obrazy_i_tokeny.png',dpi=150);plt.close(fig)
    # Matched-memory comparison is more relevant than oscillator versus AR1.
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    labels=[];gains=[];cis=[]
    for im in range(3):
        for group in ['logit','box']:
            r=next(r for r in ph['evaluation'] if r['image']==im and r['group']==group and r['model']=='oscillator')
            labels.append(f'Obraz {im+1}\n'+('logit' if group=='logit' else 'ramka'))
            gains.append(r['gain_vs_ar2_pct']);cis.append(r['ci_vs_ar2_pct'])
    ax.bar(range(6),gains,color=['#0072B2','#D55E00']*3)
    for i,ci in enumerate(cis):ax.plot([i,i],ci,color='black')
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(6),xticklabels=labels,ylabel='Zmniejszenie MSE [%]',
        title='Oscylator względem zwykłego AR(2): taka sama długość pamięci')
    fig.savefig(OUT/'faza_vs_taka_sama_pamiec.png',dpi=160);plt.close(fig)
    # Audit provenance; hash originals read-only, record all checkpoint file metadata.
    sources=[];weightdir=Path(ext['source_weight_dir'])
    for k in d['steps']:
        f=weightdir/f'k{int(k):05d}.pt';st=f.stat()
        row=dict(step=int(k),path=str(f),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        if int(k) in [0,33300,65640]:row['sha256']=sha(f)
        sources.append(row)
    manifest=dict(source_checkpoints=sources,image_cache=ext['names'],image_sha256=ext['image_hashes'],
        source_code_sha256={str(f.relative_to(ROOT)):sha(f) for f in CODE.glob('*.py')},
        original_model_sha256=sha(ROOT/'experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py'),
        quality_checks='passed: finite data, unique selected tokens, disjoint masks, no third-image fitting, direct residual identity, finite difference, full-rank reconstruction')
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    def add(s=''):lines.append(s)
    add('# Próba analogii światła na zapisanych checkpointach SmallViT')
    add('\nData: 15 października 2026. Wyniki nowych obliczeń na istniejącym biegu s47. Aplikacja i zapisane wagi nie były modyfikowane; nie uruchamiano treningu ViT.')
    add('\n**Wniosek:** opis superpozycji pozwala mierzyć rzeczywiste wzmacnianie i kompensację wkładów neuronów. Nie uzyskano jednak stabilnej przewagi prognostycznej cech „optycznych” nad kontrolami. Pamięć dwóch poprzednich zmian pomaga prognozować wyjście, lecz zapis tłumionego oscylatora nie ma wyraźnej przewagi nad zwykłym AR(2). Nie jest to dowód fazy optycznej ani mechanizmu światła w modelu.')
    add('\n## 1. Co faktycznie policzono')
    add('\n- 806 checkpointów: kroki 0–65 640; trzy różne obrazy z istniejących historii aplikacji, łącznie 2418 przejść obraz–checkpoint.')
    add('- 24 stałe tokeny na obraz, razem 58 032 obserwacje token–checkpoint. Po 8 przy środku anotacji, 8 przy brzegu, 8 w tle. To grupy geometryczne, nie maski segmentacji ani wyłącznie dodatnie komórki straty.')
    add('- Pełny forward na RTX 3070, float32, eval. Odtworzono także dokładne pochodne końcowej głowy. Wagi i bufory BatchNorm z każdego checkpointu wczytywane strict=True.')
    add('- Analiza dotyczy ostatniego MLP: z ma 768 współczynników, a u_c=z_c v_c działa w 192-wymiarowym strumieniu residual. Ruch między checkpointami mierzono w tym rzeczywistym strumieniu; nie w rzutowanym filmie 3D.')
    add('- Dla prognoz użyto 646 regularnych zapisów od 1100 do 65 600 co 100 aktualizacji. Dodatkowych końców epok nie interpolowano ani nie mieszano z odstępem 100.')
    add('- Obraz 3 nie uczestniczył w dopasowaniu ani wyborze hiperparametrów modeli prognostycznych. Nie ustalano, czy pochodzi z innego filmu. Wszystkie trzy historie dotyczą tego samego treningu s47, więc nie są trzema niezależnymi biegami.')
    add(f'\n![Obrazy i wybór tokenów]({link(OUT/"obrazy_i_tokeny.png")})')
    add('\n## 2. Rzeczywista superpozycja i „poziom bez współdziałania”')
    add(r'''
\[
U=\sum_c u_c,\qquad P_{\rm diag}=\sum_c\|u_c\|^2,\qquad
P_{\rm net}=\|U\|^2,\qquad
\eta=P_{\rm net}/P_{\rm diag}.
\]
\[
P_{\rm net}-P_{\rm diag}=2\sum_{c<d}u_c^\top u_d.
\]

η>1 oznacza dodatnią sumę wyrazów mieszanych, η<1 — ujemną. P_diag jest również oczekiwaną normą kwadratową sumy przy hipotetycznych niezależnych, jednostajnie losowanych fazach w zespolonym rozszerzeniu tych wektorów. Nie oznacza to, że takie fazy istnieją w sieci. Wielkości nie są mocą w watach ani energią. Bias MLP wyłączono z sumy neuronów i zachowano osobno przy odtwarzaniu rzeczywistego kroku.

Nie mylić η z q=||sum u||/sum ||u||: q≤1 również wtedy, gdy η>1. Mały stosunek do sumy długości nie dowodzi wygaszania względem odniesienia P_diag.
''')
    add('| Obraz | Mediana η przy środku, inicjalizacja | Mediana η przy środku, kroki ≥55 000 |')
    add('|---|---:|---:|')
    for im in range(3):
        v=[next(r for r in desc if r['image']==im and r['role']==0 and r['window']==w)['ratio_power']['median'] for w in ['initial','late']]
        add(f'| {im+1} | {v[0]:.3f} | {v[1]:.3f} |')
    add('\nW późnym oknie wypadkowa norma kwadratowa jest więc przeciętnie większa od sumy norm kwadratowych pojedynczych wkładów. Wzrost nie jest monotoniczny: krzywe najpierw rosną, później częściowo maleją. To pomiar organizacji wkładów; bez interwencji nie dowodzi, że ona powoduje lepszą detekcję.')
    add('\nDodatkowo policzono odpowiednik podatności: J głowy dla logitu i czterech krawędzi znormalizowanych do wymiarów obrazu. Te same ilorazy zastosowano do J u_c. Wynik zależy od kierunku odczytu: znoszenie dla logitu i dla geometrii ramki nie musi być takie samo. Jest to liniowa odpowiedź przy rzeczywistym stanie po MLP, nie efekt skończonego usunięcia neuronu.')
    add('\n## 3. Czy te cechy pomagają przewidzieć następny checkpoint?')
    add('\nDopasowanie obejmuje wyłącznie obrazy 1–2 i cele przed krokiem 33 400; walidacja zaczyna się od 33 400, test od 43 000. Jednokrokowy horyzont wynosi 100 aktualizacji. W teście za każdym razem dostępny jest rzeczywisty obecny stan; nie jest to swobodna symulacja całej przyszłej trajektorii.')
    add(f'\nLiczby wierszy: dopasowanie {pred["split"]["train_rows"]}, walidacja {pred["split"]["validation_rows"]}, test {pred["split"]["test_rows"]}. W części testowej jest 226 momentów czasu na obraz. PCA, standaryzacje i parametry dopasowano bez obrazu 3 i bez przyszłych checkpointów.')
    add(f'\nBaseline ma {pred["base_feature_count"]} cech: czas, normy stanu/aktywacji/wkładów, P_net i P_diag, obecne wyjście, poprzednią zmianę, zgodność wcześniejszego ruchu z U, położenie tokenu, grupę oraz po 8 składowych PCA stanu i U. Dodatek ma {pred["added_feature_count"]} cech: ilorazy superpozycji, efektywną liczbę kanałów i miary odpowiedzi głowy. Zysk samego ilorazu η byłby efektem użytecznej parametryzacji, ponieważ jego składniki są już w baseline.')
    add('\nPorównano ridge (alpha wybierane na walidacji) i nieliniowy histogram gradient boosting o ustalonej konfiguracji. Kontrola tasuje dodatkowe cechy między tokenami wewnątrz tego samego obrazu, czasu i grupy, zachowując ogólny trend. Trening tych małych modeli analitycznych nie zmienia ViT.')
    add('\n**Trzeci obraz, późniejsze checkpointy. Dodatnie wartości to zmniejszenie MSE; ujemne to pogorszenie.**')
    add('\n| Cel | Ridge + cechy optyczne | Nieliniowy + cechy optyczne | Przedział 95% dla nieliniowego | Nieliniowy + tasowane cechy |')
    add('|---|---:|---:|---:|---:|')
    names={'latent_all':'Długość ruchu latentu, 24 tokeny','logit_center':'Zmiana logitu tokenu środkowego','box_center':'Zmiana czterech krawędzi tokenu środkowego'}
    for group,label in names.items():
        rr={r['model']:r for r in pred['evaluation'] if r['image']==2 and r['group']==group}
        a=rr['hgb_optical'];ci=a['ci_vs_base_pct']
        add(f'| {label} | {rr["ridge_optical"]["gain_vs_base_pct"]:+.2f}% | {a["gain_vs_base_pct"]:+.2f}% | [{ci[0]:+.2f}; {ci[1]:+.2f}]% | {rr["hgb_shuffled_optical"]["gain_vs_base_pct"]:+.2f}% |')
    add('\nDla ramki trzeciego obrazu RMSE prognozy krawędzi spadło z 9,59 do 9,45 piksela. Jednak na obrazach 1 i 2 ten sam dodatek pogorszył prognozę ramki odpowiednio o 4,82% i 2,22% MSE. Nie ma stabilnego zysku między obrazami i rodzinami modeli. Cechy mogą być użyteczne lokalnie, ale wynik nie uzasadnia ogólnej tezy o lepszym modelu dynamiki.')
    add('\n**Te liczby dotyczą prognozowania zmian wyjścia zamrożonych checkpointów. Nie są poprawą IoU, mAP ani jakości oryginalnego detektora.** Bbox jest odczytem ustalonego tokenu najbliższego środkowi anotacji: 226, 317 i 369. Nie jest to końcowy wybór detekcji po argmax/NMS.')
    add(f'\n![Superpozycja i test prognozy]({link(OUT/"superpozycja_i_prognoza.png")})')
    add('\n## 4. Próba fazy: oscylator kontra zwykła pamięć')
    add(r'''
Dla zmian odczytu d_n=y_n−y_(n−1) dopasowano:

\[
d_{n+1}=c+a d_n+b d_{n-1}.
\]

W modelu oscylacyjnym narzucono a=2r cos(ω), b=−r², 0<r<1. Zwykły AR(2) ma swobodne a,b. Parametry są wspólne dla dwóch obrazów dopasowania, osobne dla pięciu odczytów i trzech grup tokenów: 15 modeli. Stabilny oscylator przeszukuje z góry zadaną siatkę tłumienia i okresów; nie używa transformaty Hilberta ani przyszłości do wyznaczania fazy.
''')
    add('\n| Trzeci obraz | Zysk oscylatora względem AR(1) | Zysk względem AR(2) | Przedział 95% względem AR(2) |')
    add('|---|---:|---:|---:|')
    for group in ['logit','box']:
        r=next(r for r in ph['evaluation'] if r['image']==2 and r['group']==group and r['model']=='oscillator');ci=r['ci_vs_ar2_pct']
        add(f'| {group} | {r["gain_vs_ar1_pct"]:+.2f}% | {r["gain_vs_ar2_pct"]:+.2f}% | [{ci[0]:+.2f}; {ci[1]:+.2f}]% |')
    add('\nPamięć dwóch zmian daje więc rzeczywisty zysk względem krótszej pamięci w tej prognozie. Specyficzny zapis oscylacyjny daje względem zwykłego AR(2) mniej niż 1% na trzecim obrazie, a wszystkie sześć przedziałów dla porównania oscylator–AR(2), na trzech obrazach i dwóch odczytach, obejmują zero.')
    add('\n**Kontrola pozornej fazy.** Wszystkie 15 modeli AR(2) zmian miały zespolone bieguny, odpowiadające okresom około 271–309 kroków. Jednak w 200/200 symulacji całkowicie niezależnego szumu położeń jego różnice również dały zespolone bieguny. Powód można wyprowadzić bez fizyki światła: dla y_n=ε_n z niezależnym szumem różnice mają ujemną autokorelację dla opóźnienia jeden. Najlepszy liniowy AR(2) ma wtedy a=−2/3, b=−1/3, co daje pozorny okres około 287 kroków przy próbkowaniu co 100. Sama obecność zespolonych biegunów nie identyfikuje oscylatora fizycznego.')
    add('\nKontrola na surowych odczytach, bez różnicowania, dała 0/15 modeli z zespolonymi biegunami. Wymuszenie oscylatora pogorszyło prognozę względem AR(2). Modele poziomów miały także problem przeniesienia średniego bbox między obrazami; dlatego ich porażka nie dowodzi braku jakichkolwiek oscylacji w sieci. Wniosek dotyczy braku wystarczających dowodów na fazę w tej konkretnej próbie.')
    add(f'\n![Porównanie przy tej samej pamięci]({link(OUT/"faza_vs_taka_sama_pamiec.png")})')
    add('\n## 5. Dosłowny zapis „amplituda + faza” pojedynczego lasera')
    add(r'''
Wykonano dodatkową próbę reprezentacji. W bazie dwóch kierunków e1,e2 przypisano każdemu neuronowi:

\[
E_c=(u_c\cdot e_1)+i(u_c\cdot e_2),\quad A_c=|E_c|,\quad \phi_c=\arg E_c.
\]

Sumowanie zespolone dokładnie odtwarza rzut sumy U na tę płaszczyznę. Baza została wyznaczona na kroku 33 300 z obrazów 1–2 i zamrożona; nie obracano jej w celu dopasowania przyszłych wyników. Kierunki wybrano z macierzy drugich momentów neuronowych wkładów. Nie jest to faza elektromagnetyczna, lecz kąt w określonym układzie współrzędnych.

Rekonstruowano ostatni przyrost MLP za pomocą 2, 8, 32, 96 i 192 kierunków, pozostawiając pełny stan wejściowy, bias i rzeczywistą głowę. Pomiar obejmuje 18 wybranych checkpointów, w tym 13 od kroku 43 000. To kontrolowana rekonstrukcja odczytu, nie retrening ani zmiana zapisanej sieci.
''')
    add('| Liczba rzeczywistych kierunków | Mediana błędu względnego U, środek obrazu 3, późne zapisy | Mediana zniekształcenia krawędzi [px] |')
    add('|---|---:|---:|')
    for rank in [2,8,32,96,192]:
        rr=[r for r in cx['rows'] if r['rank']==rank and r['image']==2 and r['step']>=43000]
        add(f'| {rank} | {100*np.median([r["relative_U_error_center"] for r in rr]):.2f}% | {np.median([r["center_box_rmse_px"] for r in rr]):.3f} |')
    add('\nPojedyncza zespolona amplituda na neuron w tej wspólnej płaszczyźnie nie wystarcza do odtworzenia pełnego ruchu. Nie dowodzi to niemożliwości każdego modelu optycznego: wielomodowe pole może mieć więcej stopni swobody. Dokładny zapis wszystkich 192 współrzędnych jako 96 liczb zespolonych jest możliwy, lecz sam w sobie tylko zmienia notację i nie dodaje mocy wyjaśniającej.')
    add('\nWyraźnie mniejszy błąd odczytu ramki niż błąd wektora U pokazuje, dlaczego trzeba oddzielać ruch reprezentacji od skutku dla zadania. Nie można oceniać poprawności całego modelu ruchu wyłącznie po względnie podobnej ramce.')
    add('\n## 6. Weryfikacja rachunku i ograniczenia')
    add(f'\n- Maksymalny względny błąd odtworzenia rzeczywistego przyrostu MLP w sprawdzonych checkpointach: {max(c["residual_relative"] for c in ext["checks"]):.3g}.')
    add(f'- Błąd tożsamości wyrazów mieszanych, kontrola float64: maksymalnie {max(c["cross_relative"] for c in ext["checks"]):.3g}.')
    add(f'- Niezależna kontrola pochodnej głowy centralnymi różnicami, float64: maksymalnie {max(c.get("head_jvp_fd_relative",0) for c in cx["checks"]):.3g}.')
    add(f'- Epsilon końcowego LayerNorm: {ext["norm_eps"]}; epsilon LayerNorm MLP: {ext["mlp_norm_eps"]}. Użyto modułów modelu, a nie historycznej ręcznej kopii głowy z innym epsilon.')
    add('- Rozbieżność aktywacji względem wcześniejszego cache float16 wyniosła do około 0,025% RMS. Nowe miary policzono ze świeżego forward float32, a nie z zaokrąglonego cache.')
    add('- Nie znaleziono NaN/Inf. Wybory tokenów są stałe; obrazy mają różne hashe. Maski dopasowania, walidacji i testu są rozłączne.')
    add('- Przedziały pochodzą z 1000 replik bootstrapu bloków po 20 checkpointów. Najpierw uśredniano błąd w obrębie checkpointu. Nie traktowano 24 tokenów jako niezależnych powtórzeń. Przedziały są warunkowe dla tych trajektorii, eksploracyjne, bez korekty wielu porównań i bez niepewności między seedami.')
    add('- Jakość prognozy mierzono dla konkretnego próbkowania co 100 kroków. Ruch szybszy od tego zegara nie jest rozdzielony. Test nie obejmuje wszystkich możliwych zjawisk falowych ani wszystkich warstw ViT.')
    add('- Nie dopisano dowolnych długości fali, polaryzowalności, równań Maxwella, temperatury ani faz uczonych przez nowy ViT. Takie zmiany wymagałyby osobnego modelu i eksperymentu.')
    add('\n## 7. Co ten wynik daje dla mikroskopu')
    add('\nWarto traktować η i jego odpowiedniki dla głowy jako diagnostykę współdziałania wkładów: pozwalają odróżnić duże indywidualne działania od ich wypadkowego skutku. Jest również uzasadnienie dla opisu pamięci i naprzemiennych korekt podczas uczenia. Na tej próbie nie ma podstaw, aby przypisać neuronom fazę lasera jako dodatkową przyczynę ruchu lub uznać model falowy za lepszy ogólny predyktor.')
    add('\nNajciekawszy dalszy kierunek publikacyjny pozostaje związany z funkcją: czy grupy neuronów wskazane przez te miary odpowiadają za konkretne korekty ramki w kontrolowanych interwencjach i na niezależnych biegach. Obecne wyniki są pilotażem i kontrolą analogii, nie potwierdzeniem takiego mechanizmu.')
    add('\n## 8. Pliki i odtwarzanie')
    add(f'\n[Protokół przed obliczeniami]({link(CODE/"PROTOKOL.md")}). Rozwinięcie kontroli surowych poziomów i dosłownego zapisu zespolonego wykonano po pierwszym wyniku, aby sprawdzić alternatywne wyjaśnienia; nie używano tego do dostrajania głównego testu prognostycznego.')
    for name,label in [('pomiar.npz','Pełne pomiary float32'),('ekstrakcja.json','Kontrole ekstrakcji'),('prognozy.json','Modele i wyniki prognostyczne'),('prognozy.npz','Cele, maski i wszystkie prognozy'),('faza.json','Modele zmian i kontrola szumu'),('faza_poziomy.json','Kontrola surowych poziomów'),('zapis_zespolony.json','Amplitudy, fazy i błędy rekonstrukcji'),('MANIFEST.json','Pochodzenie danych i hashe')]:
        add(f'- [{label}]({link(OUT/name)}).')
    add('\nPolecenia z katalogu SmallViTAblation, w istniejącym środowisku Python z PyTorch/timm/NumPy/SciPy/scikit-learn/Matplotlib:')
    add('```powershell\npython -X utf8 experiments/optyka_hipoteza_20261015/ekstrakcja.py\npython -X utf8 experiments/optyka_hipoteza_20261015/analiza.py\npython -X utf8 experiments/optyka_hipoteza_20261015/zapis_zespolony.py\npython -X utf8 experiments/optyka_hipoteza_20261015/raport.py\n```')
    add('\nOdtwarzanie zapisuje pliki wyłącznie w katalogu wyników tej próby. Dla nowego wariantu należy nadać oddzielny katalog, aby zachować ten wynik.')
    add(f'\nPunkt odniesienia literaturowy: [raport wzorów i pomiarów]({link(ROOT/"outputs/research_wzory_pulapki_optyczne_20261015/ATLAS_WZOROW_I_POMIAROW.md")}). Wyniki tej próby są własnymi obliczeniami, a nie wnioskami zaczerpniętymi z tamtych artykułów.')
    (OUT/'RAPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('QA passed; saved',OUT/'RAPORT.md',flush=True)

if __name__=='__main__':main()
