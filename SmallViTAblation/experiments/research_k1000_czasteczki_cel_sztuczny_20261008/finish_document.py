from pathlib import Path
import json,re,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/Path(__file__).parent.name
p=OUT/'RAPORT.md';s=p.read_text(encoding='utf-8')
s=s.replace('Dla oracle pełna znajomość z, GT, m,v,k i reguły celu wystarcza',
    'Dla oracle pełna znajomość z, z0, GT, stałego kontekstu końcówki, m, v, k i stanu przełączenia celu wystarcza')
s=s.replace('Nie jest to procentem poprawnych predykcji, AP, dokładność ruchu',
    'Nie oznacza to procentu poprawnych predykcji, AP ani dokładności ruchu')
s=s.replace('ani korzyść doboru danych.','ani korzyści doboru danych.')
s=s.replace('Porównać mean 600×768, mean global, conditional ridge/MLP/adaptor, regime-gated shrinkage, zero correction i per-image oracle. Conditional regression bez klastrów sprawdza, czy discretization regimes wnosi cokolwiek.',
    'Porównać średnią 600×768, średnią globalną, ridge/MLP/adapter z kontekstem, korektę z reżimami i shrinkage, brak korekty oraz oracle per obraz. Regresja bez klastrów sprawdza, czy dyskretyzacja reżimów wnosi dodatkową wartość.')
s=s.replace('Full768','Pełne 768D')
s=s.replace('damping i memory','tłumienia i pamięci').replace('Stochastyczny term','Składnik stochastyczny')
s=s.replace('termodynamiczny heat bath','termodynamiczną kąpiel cieplną').replace('conditional variance, tails','warunkowej wariancji, ogonów rozkładu')
s=s.replace('GLE lag coefficients mogą aproksymować memory; bez wyprowadzenia/projection formalism',
    'Współczynniki opóźnień GLE mogą przybliżać pamięć; bez wyprowadzenia w formalizmie projekcji')
s=s.replace('future minibatch','przyszła paczka').replace('future gradients','przyszłych gradientów')
s=s.replace('Ten sam przyszła paczka nie daje tych samych przyszłych gradientów','Ta sama przyszła paczka nie daje tych samych przyszłych gradientów')
s=s.replace('zgradientami','z gradientami')
s=s.replace('zachowanie/teacher reference','zachowanie lub odniesienie do nauczyciela')
s=s.replace('pooled 95,4','wynikiem 95,4 dla reprezentacji po poolingu')
s=s.replace('poolingu tokenów narzędzia','poolingu tokenów regionu narzędzia')
s=s.replace('LR.05,101–250 .15,251–400 .05,401–500 .01','LR 0,05; 101–250: 0,15; 251–400: 0,05; 401–500: 0,01')
s=s.replace('oracle:1–100','oracle: 1–100')
s=s.replace('Regime label może przewidywać optimizer phase albo baseline difficulty.',
    'Etykieta reżimu może przewidywać fazę optimizera albo początkową trudność obrazu.')
s=s.replace('LR transitions','zmian LR').replace('image-dependent','zależny od obrazu')
s=s.replace('Average after delta nie jest average stanu w tej samej iteracji.',
    'Średnia delta after nie jest średnią stanu w tej samej iteracji.')
s=s.replace('clustering na widocznym ruchu W2delta versus raw 768','klasteryzacja widocznego ruchu W2 delta kontra pełne 768D')
s=s.replace('stability klastrów across seeds i filmów','stabilność klastrów między seedami i filmami')
s=s.replace('prediction z samego k, LR, fazy celu, normy i target position jako baseline',
    'predykcja z samego k, LR, fazy celu, normy i pozycji celu jako kontrola')
s=s.replace('strata walidacyjna guard','kontrola straty walidacyjnej')
s=s.replace('To dwa różne pomiary. S_nowa','To dwa różne pomiary. S_nowa')
s=s.replace('Oddzielić stary i zaktualizowany BN, a wynik końcowy oceniać w takim trybie jak deployment.',
    'Oddzielić stary i zaktualizowany BN, a wynik końcowy oceniać w takim trybie jak deployment. W pełnym trainerze aktualizacja buforów BN wnosi dodatkowy składnik do zmiany z; sam J_theta Delta theta go nie obejmuje. Należy zamrozić BN w kontroli albo osobno uwzględnić zmianę buforów, a potem zmierzyć cały krok.')
anchor='### 4.2. Nullspace fc2: co wolno interpretować'
extra='''### 4.1a. Jak ograniczyć oracle bez udawania pełnej osiągalności

[TEST PROPONOWANY] Porównać trzy parametryzacje: swobodne z768, z ograniczone koniecznym box i normą oraz z=GELU(fc1(norm2(h192))) z optymalizowanym h192. Ostatnia zachowuje lokalną mapę norm2/fc1 i jej sprzężone współrzędne. Nadal nie gwarantuje, że h każdego tokenu może powstać z jednego obrazu ani że jest zgodny z residual. To silniejsza relaksacja, nie dowód członkostwa w naturalnej rozmaitości.

Cel do transferu warto definiować jako zbiór: poprawna geometria, ranking i ograniczone false positives przy minimalnym widocznym ruchu. Wybranie jednego arbitralnego wektora z tego zbioru tworzy niepotrzebny problem inwersji. Kilka startów oracle pozwala zmierzyć, czy podobny wynik detekcji wymaga podobnego kierunku, czy tylko dowolnej reprezentacji równoważnej dla głowy.

'''
s=s.replace(anchor,extra+anchor)
anchor='### 8.3. Langevin i pamięć'
extra='''[WNIOSEK] Skończony model Koopmana wymaga, aby wybrane obserwable miały domkniętą ewolucję w rozpatrywanym zakresie; duża zgodność regresji tego nie dowodzi. Praktyczny test niedomknięcia: znaleźć stany o podobnym z i prędkości, ale różnych m/v, gradientach lub BN, a następnie porównać ich przyszłe przyrosty przy tej samej paczce. Systematycznie różne przyszłości obalają wystarczalność takiego stanu zredukowanego. Dokładać zmienne wtedy, gdy poprawiają prognozę na odłożonych trajektoriach, a nie tylko dopasowanie historii.

'''
s=s.replace(anchor,extra+anchor)
figure=f'''\n![Audyt historycznego celu K1000]({(OUT/'AUDYT_OSIAGALNOSCI.png').as_posix()})

Wykres: maksima po 600 tokenach w każdym kanale kontra konieczna granica oraz rozkład energii delty. Dotyczy jednego historycznego celu J128. Źródło: własny audyt CPU i jego zapisane hashe. [Wersja do eksportu PDF]({(OUT/'AUDYT_OSIAGALNOSCI.pdf').as_posix()}).

'''
s=s.replace('### Etykiety epistemiczne',figure+'### Etykiety epistemiczne')
# Zwarta lista: jedna pusta linia przed listą, bez sztucznego rozdzielania wszystkich elementów.
s=re.sub(r'(?m)(^[^-\n].*?\n|^- .*?\n)\n(?=(?:- |\d+\. ))',r'\1',s) if False else s
s=re.sub(r'(?m)(^\d+\. [^\n]+|^- [^\n]+)\n\n(?=(?:\d+\. |^- ))',r'\1\n',s)
# Wszystkie linki do własnych dokumentów są rzeczywistymi ścieżkami absolutnymi.
for name in ['BIBLIOGRAFIA.md','CLAIMS_EVIDENCE_FAILTEST.md','REKOMENDACJE.md']:
    s=s.replace('- '+name+':','- ['+name+']('+ (OUT/name).as_posix()+'):')
p.write_text(s.rstrip()+'\n',encoding='utf-8')

p=OUT/'BIBLIOGRAFIA.md';s=p.read_text(encoding='utf-8')
for line in list(s.splitlines()):
    if line.startswith('| R'):
        fields=line.split('|');fn=fields[-2].strip()
        if fn.endswith('.pdf'):
            fields[-2]=' [PDF]('+ (OUT/'literature'/fn).as_posix()+') '
            s=s.replace(line,'|'.join(fields))
s=s.replace('R17: dokładna surjektywność','R17: brak surjektywności')
s=s.replace('R15/R16: optimizer-aware, skończony horyzont attribution daje perspektywę analityczną,',
    'R15/R16: atrybucja uwzględniająca optimizer i skończony horyzont daje perspektywę analityczną,')
s=s.replace('unique inverse','jednoznacznej inwersji').replace('sztucznego code','sztucznego kodu')
s=s.replace('PDF s.6: §4, równanie 8, PBRF;PDF s.2','PDF s.6: §4, równanie 8, PBRF; s.2')
p.write_text(s.rstrip()+'\n',encoding='utf-8')

for name in ['CLAIMS_EVIDENCE_FAILTEST.md','REKOMENDACJE.md']:
    p=OUT/name;s=p.read_text(encoding='utf-8')
    for fn in ['RAPORT.md','BIBLIOGRAFIA.md','CLAIMS_EVIDENCE_FAILTEST.md']:
        s=s.replace(fn,'['+fn+']('+ (OUT/fn).as_posix()+')')
    p.write_text(s,encoding='utf-8')
print('Zapisano finalną wersję z wykresem, testem domknięcia i parametryzacjami oracle.')
