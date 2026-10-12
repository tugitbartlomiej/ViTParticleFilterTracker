from pathlib import Path
import re
OUT=Path(__file__).resolve().parents[2]/'outputs'/Path(__file__).parent.name
p=OUT/'RAPORT.md'
text=p.read_text(encoding='utf-8')
for bad,good in {
 'fc 1':'fc1','fc 2':'fc2','norm 2':'norm2','z 0':'z0','q 0':'q0','r 0':'r0','b 2':'b2',
 'q+c 1':'q+c1','h 192':'h192','beta 1':'beta1','beta 2':'beta2','DINOv 2':'DINOv2','cp 1250':'cp1250',
 'compatible_k 8000_target':'compatible_k8000_target','finalnormalized state':'stan po końcowym LayerNorm',
 'Przeniesienie rowspace preserves W2 delta.':'Rzut na rowspace zachowuje W2 delta.',
 'smallest singular value 0,2920, largest 1,1364; condition≈3,8915':'najmniejsza wartość singularna 0,2920, największa 1,1364; liczba uwarunkowania około 3,8915',
 'Nie tylko abstrakty.':'Analiza obejmowała treść artykułów, nie tylko abstrakty.',
 'Przejście granic jest konieczne, ale niewystarczające.':'Spełnienie granic jest konieczne, ale niewystarczające.',
 'pooled representation':'reprezentacja po poolingu',
 'domination przez outliers':'dominację obserwacji odstających',
 'Average suppression tła może pomagać na distribution':'Średnie tłumienie tła może pomagać na rozkładzie',
 'Averaging delty':'Uśrednianie delty','systematic bias':'systematyczny bias',
 'W każdej fold:':'W każdym podziale walidacji:',
 'scaler, feature selection, clustering, liczba klastrów i korekty fit tylko na train films;':'skalowanie, dobór cech, klasteryzacja, liczba klastrów i korekty dopasowane tylko na filmach train;',
 'model przypisania reżimu fit z natural z0/context, bez held-out oracle target;':'model przypisania reżimu uczony z naturalnego z0 i kontekstu, bez celu oracle z danych odłożonych;',
 'wybór hyperparameters na osobnej validation; końcowy test untouched;':'dobór hiperparametrów na osobnej walidacji; końcowy test nietknięty;',
 'miarą podstawową jest head utility i generalizacja, nie agreement z klastrem oracle.':'podstawową miarą jest użyteczność głowy i generalizacja, nie zgodność z klastrem oracle.',
 'siatkę 20×30':'siatkę 20×30','z(z0,delta,trajectory)':'z danych (z0, delta, trajectory)',
 'pierwszą i drugą chwilę':'pierwszy i drugi moment',
 'Pełny stan uczenia obejmuje wagi, pierwszy i drugi moment, step counters, BN buffers, scheduler, sampler/RNG i augmentację.':'Pełny stan uczenia obejmuje wagi, momenty AdamW, liczniki kroków, bufory BN, scheduler, sampler/RNG i augmentację.',
 'variance stanu':'wariancji stanu','variance ruchu':'wariancji ruchu','Duża static structure':'Duża struktura statyczna',
 'percentage correct predictions':'procentem poprawnych predykcji','zbaseline':'z baseline',
 'Denominator bardzo mały':'Bardzo mały mianownik','niezależnych trajectories':'niezależnych trajektorii',
 'raw coordinates':'surowych współrzędnych','grid ordering':'kolejność siatki',
 'naturalny head outcome':'naturalny wynik głowy','equivalent behavior':'równoważne zachowanie',
 'exact raw match':'dokładnego dopasowania surowych współrzędnych',
 'wyjście docelowe może nadal zostać zrealizowany fooling image':'wyjście docelowe może nadal zostać zrealizowane przez obraz oszukujący model',
 'independent realism i semantic constraints':'niezależne kryteria realizmu i semantyki',
 'lub patient':'lub pacjenta','DINO neighbors':'Sąsiedzi DINO','Synthetic query artifacts':'Artefakty syntetycznego zapytania',
 'TRAIN-only 35 000':'35 000 wyłącznie TRAIN','targetlearning':'uczenia w stronę celu',
 'Dalszy trening control sam znacznie pomagał; effects trzeba liczyć ponad niego.':'Dalszy trening kontrolny sam znacznie pomagał; efekt wariantu trzeba liczyć względem niego.',
 'Independent per-image oracle':'Niezależny oracle per obraz','relative błąd Taylora':'względny błąd Taylora',
 'Raw odległość':'Surowa odległość','authentic continuation':'autentycznej kontynuacji',
 'BNold i BNupdated':'stary i zaktualizowany BN','strata walidacyjna guard':'kontrola straty walidacyjnej',
 'Denominator zależy':'Mianownik zależy','SGD guarantees nie są guarantees AdamW':'gwarancje dla SGD nie są gwarancjami AdamW',
 'weighted subset gradient':'gradient ważonego podzbioru','prawdziwe parameter-learning trajectories':'rzeczywiste trajektorie uczenia parametrów',
 'hidden states':'stanów ukrytych','na test accuracy':'pod względem trafności na teście',
 'strata detekcji, AP':'strata detekcji, AP',
 'test:80 tworzy expected trajectory float16':'test_pelne_latenty.py:80 tworzy oczekiwaną trajektorię float16',
 'To finding wersji z hash zapisanym w evidence; inny agent może je później poprawić.':'To ustalenie dla migawki z hashem zapisanym w evidence; inny agent może później poprawić kod.',
 'przez target/tło, filmy, difficulty, magnitude i fazę LR':'według narzędzia/tła, filmów, trudności, wielkości ruchu i fazy LR',
 'czyli 1,385TB decimal':'czyli 1,385 TB w zapisie dziesiętnym','byłoby≈2,364TB':'byłoby około 2,364 TB',
 'panel≈18,165GB':'panel około 18,165 GB','trajectory jako float16':'trajektorię jako float16',
 'kwantyzacji trajectory':'kwantyzacji trajektorii',
}.items(): text=text.replace(bad,good)

def section(start,end,new):
    global text
    a=text.index(start);b=text.index(end,a)
    text=text[:a]+new.rstrip()+'\n\n'+text[b:]

section('## 12.','## 13.', '''## 12. Minimalny program falsyfikacji przed kosztownymi obliczeniami

Wszystkie poniższe testy są propozycjami. Nie wykonano ich w tym researchu. Panel 32–64 reprezentatywnych obrazów jest pilotażem wykrywającym duże błędy, nie finalną próbą do twierdzeń o generalizacji.

| Bramka | Minimalny test | Wynik zatrzymujący dalszą inwestycję | Co można twierdzić po przejściu |
|---|---|---|---|
| G0: aparat | Hash, zgodność hook/tail, kształt i kolejność tokenów, GT/decode; granice norm2/fc1 | Niezgodny hash, cache z innej rodziny, cel poza koniecznymi granicami | Aparat mierzy właściwy system |
| G1: oracle | 32–64 obrazy; swobodne 768D kontra ograniczenia naturalnej mapy i minimalny widoczny ruch | Sukces zależy od nieosiągalnych współrzędnych lub zanika po ograniczeniach | Użyteczna interwencja istnieje w testowanym panelu |
| G2: bez GT | Podział po filmach; średnia, reżimy, regresja, brak korekty; AP/FP z pełnego forwardu | Brak przewagi nad zerem, losowym kierunkiem lub prostym adapterem | Korektor generalizuje w zakresie testowanych filmów |
| G3: obraz | Kontrola naturalnego preimage; wiele startów inwersji obrazu/SD; residual i głowa | Twarda dolna granica błędu, oszukiwanie głowy lub brak niezależnej zgodności semantycznej | Generator realizuje część zgodnych celów |
| G4: wspólne wagi | Projekcja celu JVP/VJP i krótkie kroki na kopiach pełnego stanu | Duża reszta projekcji, błędny znak efektu lub pogorszenie walidacji | Część celu jest lokalnie osiągalna przez uczenie |
| G5: paczki | 8–16 paczek: random, DINO, gradient; ranking przed kontynuacjami 1/5/25/50 | Ranking nie przewyższa random lub występuje uboczna strata AP | Selector przewiduje użyteczność w danym stanie |
| G6: wartość SD | SD→DINO kontra original→DINO i bezpośredni gradient przy wspólnym budżecie | SD nie poprawia zależności jakości od kosztu | SD wnosi mierzalną wartość w tym protokole |
| G7: generalizacja | Co najmniej 3 seedy, kilka pul i stanów, nowe filmy, wszystkie koszty | Przedział ufności korzyści obejmuje zero lub czas do tego samego AP jest gorszy | Ograniczone, odtwarzalne twierdzenie empiryczne |

Kryteria zamrozić przed pomiarami. Główna metryka: AP na nowych filmach; dodatkowo FP przy ustalonym punkcie pracy, recall i kalibracja. Raportować sparowany bootstrap po filmach oraz czas do tego samego AP. Nie uznawać arbitralnego 0,95 za próg potwierdzający teorię. W G5 podać Spearman/Kendall rank correlation, top-k regret i realizowaną użyteczność z przedziałami ufności. Jeśli 8–16 paczek nie wystarczy do rozstrzygnięcia, rozszerzyć próbę przed dużym biegiem.

Tolerancję inwersji wyznaczyć z kontroli naturalny→naturalny cel i precyzji numerycznej. Porównania mają te same augmentacje, optimizer, schedule i liczbę aktualizacji; dodatkowo wymagają osobnego wariantu przy równym czasie całkowitym. Obliczanie gradientów wielu kandydatów może kosztować więcej niż zwykły trening. Walidacja używana przez selector nie może pełnić roli końcowego testu.

### Sześć konkretnych hipotez

**H1.** Cele z ograniczeniami naturalnej mapy i minimalną składową null będą łatwiejsze do inwersji i transferu, nawet jeśli zmniejszą odsetek sukcesów oracle.
Test obalenia: przy dopasowanej normie i budżecie nie poprawiają błędu osiągalności ani użyteczności na nowych filmach. Sam rzut rowspace nie wystarcza; cel ponownie rozwiązać z ograniczeniami.

**H2.** Korektor zależny od kontekstu przewyższy średnią przestrzenną, ponieważ kontekst determinuje znak ruchu.
Test obalenia: po podziale po filmach brak przyrostu ponad stały adapter, a przypisanie klastrów oracle z z0 jest na poziomie losowym.

**H3.** Reżimy semantyczne przetrwają zmianę schedule i będą przewidywać niezależną semantykę na danych odłożonych.
Test obalenia: klastry wyjaśnia głównie k/LR/faza sukcesu, a zmiana optimizera usuwa ich stabilność i użyteczność.

**H4.** Score Q_B uwzględniający stan AdamW przewidzi rzeczywisty ruch H1 i ranking użyteczności lepiej niż odległość DINO.
Test obalenia: sparowane próby kandydatów nie wykazują lepszego rankingu lub mniejszego regret niż random/DINO, także w przestrzeni widocznej dla zadania.

**H5.** Użyteczność H1 jest dostatecznie trwała, aby pomagać selekcji danych.
Test obalenia: efekt znika lub zmienia znak po 25–50 wspólnych przyszłych krokach.

**H6.** SD dostarcza informacji ponad retrieval naturalnych obrazów przy równym koszcie.
Test obalenia: dobór z tej samej puli bez SD osiąga taką samą lub lepszą jakość i czas do docelowego AP.
''')

section('## 14.','## 15.', '''## 14. Co zmienić w pipeline

Rekomendowana kolejność:

1. Naturalny K1000, GT, zgodny hook i pełny kontekst residual.
2. Mały zestaw oracle z koniecznymi granicami, ograniczeniem normy i rozkładem visible/null.
3. Korektor zależny od kontekstu bez GT albo adapter uczony nadzorem pośrednim i stratą detekcji.
4. Cel zachowania głowy lub widocznego ruchu, sprawdzony pod kątem lokalnej osiągalności przez wspólne wagi.
5. Pula prawdziwych kandydatów TRAIN z kontrolą różnorodności; SD jako opcjonalne rozszerzenie.
6. Rzeczywista aktualizacja kandydata uwzględniająca AdamW oraz JVP/pomiar na sondach, z kontrolą walidacji.
7. Krótkie sparowane kontynuacje, a dopiero po nich większa selekcja.
8. Nowe filmy, kilka seedów i porównanie jakości względem całkowitego kosztu.

| Wariant | Argument za | Główne ryzyko | Rekomendacja |
|---|---|---|---|
| Pełny oracle 768D→reżimy→SD→DINO→Langevin | Bogata diagnoza i wizualizacja | Niezweryfikowane mosty, nullspace, niewykonalne cele, koszt | Rozłożyć na osobne bramki badawcze |
| Korektor lub nadzór inspirowany LabelEnc | Bezpośredni transfer etykiet do uczonego modelu | Bias celu, współadaptacja głowy, etap pomocniczy | Pierwsza tania kontrola transferu |
| Selekcja według walidacji i stanu optimizera | Score bliższy deklarowanej korzyści uczenia | Koszt gradientów, krótki horyzont, przeuczenie walidacji | Pierwsza kontrola selekcji |
| Retrieval prawdziwych obrazów z naturalnych zapytań | Prosty i relatywnie tani | Podobieństwo nie równa się użyteczności, tło, duplikaty | Obowiązkowy baseline |
| Dopasowanie gradientów lub trajektorii wag | Literatura wiąże dane z rzeczywistym procesem uczenia | Koszt rozwijania wielu kroków, zależność od architektury/stanu | Po pozytywnym teście lokalnego transferu |

Terminologia cząsteczek pozostaje użyteczna do opisu. Opis ruchu we wspólnej bazie trzeba oddzielać od interpretacji mechanizmu; 600 tokenów, 768 osi MLP i 192 współrzędne residual pełnią różne role.
''')

section('## 15.','Dokumenty pomocnicze:', '''## 15. Zakres zakończenia i ograniczenia

Zakończono analizę, pełnotekstowy przegląd kluczowych prac, audyt CPU checkpointa, archiwizację dowodów oraz opracowanie falsyfikowalnych hipotez i testów. Walidacja empiryczna całego łańcucha pozostaje do wykonania; proponowane forwardy, treningi, kontynuacje i generacja SD nie zostały uruchomione.

Zapisy wykonano wyłącznie w nowych katalogach tego researchu. Cudze eksperymenty pozostały nietknięte; nie uruchomiono GPU ani jobów, nie wykonano commita. Repozytorium było współdzielone i zawierało istniejące zmiany. Późniejsze wyniki i poprawki innych wykonawców należy porównywać z hashami zachowanych migawek.
''')

# Doprecyzowanie zweryfikowanych stron i dwóch rozróżnień modelu.
text=text.replace('PDF s.5–8, eq.7–10','PDF s.5–8, równania 7–9 na s.6')
text=text.replace('§3–4, dowodzą non-surjectivity','PDF s.3–5, §3–4, dowodzą braku surjektywności')
text=text.replace('Algorithm 2, oraz oficjalny','PDF s.3, Algorithm 2, oraz oficjalny')
text=text.replace('eq.8, pokazują, że influence','PDF s.6, równanie 8, pokazują, że influence')
text=text.replace('stanów 0..500','stanów 0–500')
text=text.replace('-64 POD modes','- 64 mody POD').replace('-8 podmian','- 8 podmian')
text=text.replace('confidence 0,84576','confidence 0,84576').replace('K8000:IoU0','K8000: IoU 0')
text=text.replace('straty detekcji, AP','straty detekcji, AP')
text=text.replace('strata walidacyjna i AP','stratę walidacyjną i AP')
text=text.replace('falsyfikowalne tezy i testów','falsyfikowalne tezy i testy')
text=text.replace('jawnie wskazanych skillów','jawnie wskazanych umiejętności')
text=text.replace('mogą polegać','mogą polegać')

# Czytelna prezentacja matematyki bez zmiany treści dowodu.
text=text.replace('''u = (h-mean(h)1)/sqrt(var(h)+epsilon).
Wtedy 1^T u=0 i ||u||²≤d. LayerNorm daje gamma⊙u+beta.''','''
$$u=\\frac{h-\\bar h\\mathbf1}{\\sqrt{\\operatorname{var}(h)+\\epsilon}},\\qquad
\\mathbf1^T u=0,\\qquad \\|u\\|_2^2\\le d.$$

LayerNorm daje $\\gamma\\odot u+\\beta$.''')
text=text.replace('''p_j = a_j^T beta + b_j + (a_j⊙gamma)^T u.
Niech P=I-11^T/d. Z Cauchy-Schwarza:
|p_j-c_j| ≤ sqrt(d) ||P(a_j⊙gamma)||,
gdzie c_j=a_j^T beta+b_j.''','''
$$p_j=a_j^T\\beta+b_j+(a_j\\odot\\gamma)^T u.$$

Niech $P=I-\\mathbf1\\mathbf1^T/d$. Z nierówności Cauchy'ego–Schwarza:

$$|p_j-c_j|\\le\\sqrt d\\,\\|P(a_j\\odot\\gamma)\\|_2,
\\qquad c_j=a_j^T\\beta+b_j.$$''')
text=text.replace('RMSE(Z_theta(x),Z_target) ≥ ||Z_target-clip_box(Z_target)||_F / sqrt(600·768) = 2,50349.',
 '''
$$\\operatorname{RMSE}(Z_\\theta(x),Z_*)\\ge
\\frac{\\|Z_*-\\operatorname{clip}_{\\mathrm{box}}(Z_*)\\|_F}{\\sqrt{600\\cdot768}}
\\approx2{,}50349.$$''')
text=text.replace('''m+=beta1 m+(1-beta1)g_B;
v+=beta2 v+(1-beta2)g_B²;
mhat=m+/(1-beta1^s), vhat=v+/(1-beta2^s);
Delta theta_B=-eta mhat/(sqrt(vhat)+eps)-eta lambda theta.''','''
$$m^+=\\beta_1m+(1-\\beta_1)g_B,\\qquad
v^+=\\beta_2v+(1-\\beta_2)g_B^2,$$
$$\\widehat m=\\frac{m^+}{1-\\beta_1^s},\\qquad
\\widehat v=\\frac{v^+}{1-\\beta_2^s},$$
$$\\Delta\\theta_B=-\\eta\\frac{\\widehat m}{\\sqrt{\\widehat v}+\\epsilon}-\\eta\\lambda\\theta.$$

Wzór dotyczy standardowego wariantu AdamW bez AMSGrad i bez maximize; wszystkie operacje na momentach są współrzędne.''')
# CommonMark: blank line before every list.
text=re.sub(r'(?m)([^\n])\n(?=(?:- |\d+\. ))',r'\1\n\n',text)
p.write_text(text.rstrip()+'\n',encoding='utf-8')

p=OUT/'BIBLIOGRAFIA.md';text=p.read_text(encoding='utf-8')
for bad,good in {
 'fc 1':'fc1','fc 2':'fc2','norm 2':'norm2','DINOv 2':'DINOv2','iograniczone':'i ograniczone',
 'wspólnie uczoną head':'wspólnie uczoną głowę','prostym interwencją':'prostą interwencją',
 'w badanym klasyfikacji CLIP':'w badanej klasyfikacji CLIP','Artykułów niedostępnego PDF':'Artykułów z niedostępnym PDF',
 'zwłaszcza opisu algorytmu':'zwłaszcza opis algorytmu',
 'Algorithm 2 oraz §2, propositions2–3':'PDF s.3: Algorithm 2 i Proposition 2; s.4: Proposition 3',
 '§3–4:Theorems 3.1,4.2,4.4,4.5;AppendixA:assumptions':'PDF s.3–5: twierdzenia 3.1, 4.2, 4.4, 4.5; s.15–17: dowody; Appendix A: założenia',
 '§4:eq.8PBRF':'PDF s.6: §4, równanie 8, PBRF',
 'PDF5–8:eq.7–9 iFigs.1–2':'PDF s.5–8: Figs.1–2; s.6: równania 7–9',
 'C.6:failure':'PDF s.20: Appendix C.6, porażki',
 'i_single_tensor_adamw':'`_single_tensor_adamw`',
}.items():text=text.replace(bad,good)
for pattern,repl in [(r'\bPDF(?=\d)','PDF s.'),(r'\b(Table|Fig\.|Theorem|Algorithm|Appendix|Remarks)(?=\d)',r'\1 '),
                      (r'\b(CVPR|ICLR|ECCV|NeurIPS|ICCV|ICML|AAAI|preprint)(?=20)',r'\1 ')]:
    # Bibliographic fields only, never alter URLs or local names.
    new=[]
    for line in text.splitlines():
        if line.startswith('| R'):
            fields=line.split('|')
            for ix in (2,4):fields[ix]=re.sub(pattern,repl,fields[ix])
            line='|'.join(fields)
        new.append(line)
    text='\n'.join(new)
p.write_text(text.rstrip()+'\n',encoding='utf-8')
print('Zakończono redakcję językową i doprecyzowano strony pełnych tekstów.')
