"""Redakcja i kontrola wyłącznie własnych artefaktów researchu."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / Path(__file__).parent.name
replacements = '''
bazę1500=>bazę 1500
oracle dla1500=>oracle dla 1500
siatkę20=>siatkę 20
czyli600=>czyli 600
wynosi42=>wynosi 42
nadmiaru42=>nadmiaru 42
rank192=>rangę 192
value0=>value 0
largest1=>largest 1
Kernel576=>Kernel o wymiarze 576
najmniej345600=>najmniej 345 600
błąd2=>błąd 2
daje26=>daje 26
ma74=>ma 74
nie763=>nie 763
w500=>w 500
na1500=>na 1500
indeksj=>indeks j
znatural=>z natural
wobrębie1500=>w obrębie 1500
zbarierami=>z barierami
zpermutowanymi=>z permutowanymi
zpozostawieniem=>z pozostawieniem
zsamego=>z samego
targetSD=>cel SD
zkonkretnymi=>z konkretnymi
dogradientu=>do gradientu
zmechanistyczną=>z mechanistyczną
wtej=>w tej
preprintem2026=>preprintem z 2026 roku
Theorem3.1=>twierdzenie 3.1
Interpretacja95=>Interpretacja 95
i98=>i 98
zbieżącym=>z bieżącym
daje0=>daje 0
dla180=>dla 180
wszystkich900000=>wszystkich 900 000
wprawdziwych=>w prawdziwych
wtych=>w tych
Ztej=>Z tej
zsamym=>z samym
do≈=>do około 
daje≈=>daje około 
na3=>na 3
sond×=>sond × 
skill0=>skill 0
wyjaśnia98=>wyjaśnia 98
Wynik97=>Wynik 97
zpełnych=>z pełnych
sond3=>sond 3
tokenów3=>tokenów 3
iprawdziwego=>i prawdziwego
pooled95=>pooled 95
wvisible=>w visible
trajektoriach;900000=>trajektoriach; 900 000
trajectories;900000=>trajectories; 900 000
jest900000=>jest 900 000
liczba768=>liczba 768
zcache=>z cache
targetoutput=>wyjście docelowe
zkontrolami=>z kontrolami
dokompromisu=>do kompromisu
TRAIN-only35000=>TRAIN-only 35 000
globalCLS=>global CLS
zcosine=>z cosine
wDINO=>w DINO
apozbawione=>a pozbawione
Syntheticquery=>Synthetic query
toolROI=>tool ROI
doquery=>do query
SDquery=>SD query
SmallViTfeature=>SmallViT feature
gradientutility=>gradient utility
top32=>top 32
są32=>są 32
DINO2021=>DINO 2021
targetreal=>target real
kontrdowódtransferu=>kontrdowód transferu
Sesja2026=>Sesja 2026
zgodnegoK8000=>zgodnego K8000
variantcontrol=>wariant minus kontrola
ztarget=>z target
drogi0=>drogi 0
guardAP50=>guard AP50
iinna=>i inna
rodzinaK8000=>rodzina K8000
nowegoK1000=>nowego K1000
sondX=>sond X
perimage=>per-image
wvisible/task=>w visible/task
Jacobian900000=>Jacobian 900000
iiteracyjny=>i iteracyjny
leastsquares=>least squares
trustregion=>trust region
Taylorerror=>błąd Taylora
ikierunek=>i kierunek
np.fc2=>np. fc2
igłowa=>i głowa
headupdates=>aktualizacjami głowy
latentdistance=>odległość w latencie
heldoutAP=>AP na danych odłożonych
taskloss=>strata detekcji
krokiemt=>krokiem t
paczkig_B=>paczki g_B
zopt=>z opt
awynik=>a wynik
wtakim=>w takim
wtask-visible=>w task-visible
validationloss=>strata walidacyjna
Dotproduct=>Iloczyn skalarny
zdesired=>z desired
późniejszegoAP=>późniejszego AP
Perexample=>Per-example
perexample=>per-example
dogenuine=>do rzeczywistych
trainBN=>train BN
Algorithm2=>Algorithm 2
PyTorch2=>PyTorch 2
ibiascorrection=>i bias correction
updateformula=>formuły aktualizacji
wconvex=>w convex
control idcandidate=>wariant kontrolny i kandydat
Raportować1=>Raportować 1
targetvisible=>target-visible
iAP=>i AP
scoreH1=>score H1
wnieskończoność=>w nieskończoność
pairedcausal=>paired causal
validationobjective=>celu walidacyjnego
iprzybliżenia=>i przybliżenia
weightedsubset=>weighted subset
parameterlearning=>parameter-learning
hiddenstates=>hidden states
gradientproxy=>gradient proxy
zrandom=>z random
testaccuracy=>test accuracy
wtym=>w tym
panel32=>panel 32
ogeneralizacji=>o generalizacji
shapestokens=>kształty i kolejność tokenów
zinną=>z inną
taskuseful=>użyteczna dla zadania
activationintervention=>interwencja w aktywacje
wograniczonym=>w ograniczonym
bezGT=>bez GT
fullforwardAP=>AP ze zwykłego pełnego forwardu
iFP=>i FP
nochange=>brak korekty
randomdirection=>losowy kierunek
prostegoadaptera=>prostego adaptera
wramach=>w ramach
targetmatch=>dopasowanie celu
independentsemantic=>niezależna semantyczna
targetprojection=>projekcja celu
fullstate=>pełny stan
projectionresidual=>reszta projekcji
wrongsign=>błędny znak
candidatebatches=>paczki kandydackie
przed1=>przed 1
odrandom=>od random
collateralAP=>uboczna strata AP
wartośćSD=>wartość SD
directgradient=>direct gradient
qualitycost=>jakość/koszt
najmniej3=>najmniej 3
primarymetric=>metryka główna
obejmuje0=>obejmuje 0
primaryAP=>główna metryka AP
operatingpoint=>punkt pracy
bootstrapCI=>bootstrap CI
ponadrandom=>ponad random
samegoAP=>samego AP
WscreeningG5=>W screeningu G5
dodatnieCI=>dodatnie CI
irealizowany=>i realizowany
arbitralnego0=>arbitralnego 0
inversioncontrol=>kontrola inwersji
samychaugmentations=>samych augmentacji
equalwallclock=>równy czas całkowity
scoregradientowy=>score gradientowy
Selectionvalidation=>Walidacja selekcji
finaltest=>końcowy test
donatural=>do natural
nullcomponent=>składowej null
raworacle=>raw oracle
heldoututility=>użyteczności na danych odłożonych
contextualgating=>przypisanie kontekstowe
ponadconstantadapter=>ponad stały adapter
oracleclusterassignability=>przewidywalność klastra oracle
naheldout=>na danych odłożonych
successphase=>fazę sukcesu
purportedsemantic=>domniemaną semantyczną
DINOdistance=>odległość DINO
wtaskvisible=>w task-visible
pairedcandidate=>sparowane paczki kandydackie
niżrandom=>niż random
po25=>po 25
candidatepool=>pula kandydatów
bezSD=>bez SD
stanów0=>stanów 0
czyli1=>czyli 1
Densepanel256=>Gęsty panel 256
oznaczać256=>oznaczać 256
przy256=>przy 256
sparsefull+taki globalpanel=>pełne rzadkie zapisy + taki globalny panel
wRAM=>w RAM
zbiór900000=>zbiór 900 000
tylko0=>tylko 0
dynamicregimes=>reżimów dynamicznych
Densepanel=>Gęsty panel
samplingstratification=>próbkowania warstwowego
iLRphase=>i fazę LR
wfałszywe=>w fałszywe
expectedtrajectory=>expected trajectory
Testzgodności=>Test zgodności
zPLAN=>z PLAN
zhash=>z hash
wevidence=>w evidence
exactFP32trajectories=>dokładnych trajektorii FP32
wpipeline=>w pipeline
pełnyresidualcontext=>pełny kontekst residual
zbounds/trustregion=>z bounds/trust region
conditionalcorrector=>korektor zależny od kontekstu
label-guidedadapter=>adapter nadzorowany etykietami
localweightreachability=>lokalnej osiągalności przez wagi
TRAINcandidatepool=>pula kandydatów TRAIN
zrealretrieval=>z wyszukiwaniem prawdziwych obrazów
zSD=>z SD
Optimizer-awareactualcandidateupdate=>Rzeczywista aktualizacja kandydata uwzględniająca optimizer
zvalidationutilityguard=>z kontrolą użyteczności na walidacji
pairedrollouts=>sparowane kontynuacje uczenia
Freshfilms=>Nowe filmy
Full768oracle=>Pełny oracle 768D
Conditionaladapter=>Adapter zależny od kontekstu
auxiliaryloss=>strata pomocnicza
trainablemodel=>model uczony
targetbias=>bias celu
headcoadaptation=>współadaptacja głowy
Directvalidation=>Walidacja bezpośrednia
gradientcost=>koszt gradientów
validationoverfit=>przeuczenie walidacji
Realretrieval=>Wyszukiwanie prawdziwych obrazów
znaturalnychquery=>z naturalnych zapytań
tani iprosty=>tani i prosty
trajectorydistillation=>distillation trajektorii
zupdateweighttrajectory=>z trajektorią aktualizacji wag
localtransfer=>transferze lokalnym
zjęzyka=>z języka
wewspólnej=>we wspólnej
odmechanistycznej=>od mechanistycznej
odparametrów=>od parametrów
latentdimension=>wymiarów latentu
iresidualwidth=>i szerokości residual
CPUaudit=>audyt CPU
falsyfiableclaims=>falsyfikowalne tezy
falsifiableclaims=>falsyfikowalne tezy
itestów=>i testów
ihashe=>i hashe
zrepo=>z repozytorium
od1=>od 1
iTable=>i Table
iL2=>i L2
iRemarks=>i Remarks
Taylorapproximation=>przybliżenie Taylora
validationregularization=>regularyzacja walidacyjna
crossarchitecture=>cross-architecture
iin.=>i in.
DOIpreprint=>DOI preprint
inversiontransfer=>transfer inwersji
toyexample=>toy example
wwersjizapisanej=>w zapisanej wersji
perniciousdivergence=>pernicious divergence
shiftcontrols=>kontrole przesunięcia
jointstate=>łączny stan
Theorems3=>Theorems 3
AdamWinfluence=>wpływ AdamW
restrictedtheory=>teoria ograniczona założeniami
Table2kNN=>Table 2 kNN
Table5DAVIS=>Table 5 DAVIS
AppendixF=>Appendix F
costmodel=>model kosztu
AdamWcontrols=>kontrole AdamW
odetekcji=>o detekcji
futureinputs=>przyszłych wejść
actualupdate=>rzeczywista aktualizacja
SGDdotproduct=>iloczyn skalarny SGD
iograniczenia=>i ograniczenia
narawlatenttargets=>dla surowych celów latentu
poprzezwpływ=>poprzez wpływ
frozenactivationoptimization=>optymalizację aktywacji przy zamrożonych wagach
auxiliaryfeatures=>cechy pomocnicze
jointlytrained=>wspólnie uczoną
idwaetapy=>i dwa etapy
oraclepatch=>interwencją oracle
usefuldistillation=>użytecznych dla distillation
zfeaturestatistics=>z feature statistics
ipomiarami=>i pomiarami
inverseproblem=>problemu odwrotnego
approximateposterior=>przybliżonego posterior
layoutinversion=>inwersja layoutu
izachowanie=>i zachowanie
downstreamtrainingutility=>użyteczności w dalszym uczeniu
pozaamnaturalmechanism=>poza naturalny mechanizm
exactfc2kernel=>dokładny kernel fc2
stałymresidual=>stałym residual
featurevisualizations=>wizualizacje cech
sanitychecks=>kontroli poprawności
wchirurgii=>w chirurgii
finitehorizon=>skończony horyzont
itaylorerror=>i błąd Taylora
wmetadataUndermind=>w metadanych Undermind
naPDFiarXiv=>w PDF i arXiv
napierwotnejstroniearXiv=>na pierwotnej stronie arXiv
exactsurjectivity=>dokładna surjektywność
discreteprompts=>dyskretnych promptów
idensityassumptions=>i założeniach dotyczących gęstości
continuousRGB=>ciągłego RGB
retrievedreal=>retrieved real
wbadanym=>w badanym
CLIPclassification=>klasyfikacji CLIP
wdetection=>w detekcji
counterfactualquestions=>pytania kontrfaktyczne
influenceapproximations=>przybliżenia wpływu
convexregression=>wypukła regresja
alokalnyindeks=>a lokalny indeks
selectivebackprop=>selective backprop
onlinecuration=>selekcji online
reducedgradienterror=>zmniejszony błąd gradientu
testaccuracygain=>poprawy test accuracy
iimplementacja=>i implementacja
docstringalgorytmu=>opisu algorytmu
zarchiwalnego=>z archiwalnego
checkpointstate=>stan checkpointa
hitówdeepsearch=>wyników deep search
jest166=>jest 166
ztematem=>z tematem
closedPDF=>niedostępnego PDF
fulltextread=>przeczytanych pełnych tekstów
wtytule=>w tytule
pełnychPDF=>pełnych PDF
iidentyfikatorów=>i identyfikatorów
materialem=>materiałem
inputoutput=>input/output
''' 
pairs = [line.split('=>', 1) for line in replacements.strip().splitlines()]
for name in ('RAPORT.md', 'BIBLIOGRAFIA.md'):
    p = OUT / name
    text = p.read_text(encoding='utf-8')
    for bad, good in sorted(pairs, key=lambda pair: len(pair[0]), reverse=True):
        text = text.replace(bad, good)
    # Typographic spacing at numeric boundaries; do not touch paths, hashes or identifiers.
    lines = []
    for line in text.splitlines():
        if not re.search(r'https?://|SHA256|\.py|\.pt|\.npz|outputs/|experiments/|assets/', line):
            line = re.sub(r'(?<=[a-ząćęłńóśźż])(?=\d)', ' ', line)
        lines.append(line)
    p.write_text('\n'.join(lines).rstrip()+'\n', encoding='utf-8')

urls = {
1:'https://doi.org/10.1109/CVPR.2015.7299155',
2:'https://doi.org/10.1109/CVPR.2015.7298640',
3:'https://doi.org/10.1137/15M1013857',
4:'https://arxiv.org/abs/1711.05101v3',
5:'https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html',
6:'https://doi.org/10.1609/aaai.v35i9.16988',
7:'https://arxiv.org/abs/2006.05929',
8:'https://doi.org/10.1007/978-3-030-58595-2_32',
9:'https://doi.org/10.1109/CVPR42600.2020.00874',
10:'https://arxiv.org/abs/2209.14687',
11:'https://arxiv.org/abs/2106.13933',
12:'https://arxiv.org/abs/2311.17030',
13:'https://arxiv.org/abs/2511.04638',
14:'https://arxiv.org/abs/2306.04719',
15:'https://arxiv.org/abs/2608.19762v1',
16:'https://arxiv.org/abs/2605.18814v1',
17:'https://arxiv.org/abs/2604.09839',
18:'https://arxiv.org/abs/2406.05184',
19:'https://arxiv.org/abs/2209.05364',
20:'https://arxiv.org/abs/1905.13289',
21:'https://doi.org/10.1109/ICCV48922.2021.00951',
22:'https://arxiv.org/abs/2103.00123',
23:'https://doi.org/10.1109/CVPR52688.2022.01045',
24:'https://arxiv.org/abs/2312.05021',
}
p = OUT / 'RAPORT.md'
text = p.read_text(encoding='utf-8')
text = re.sub(r'\[R(\d+)\](?!\()', lambda m: f'[R{m[1]}]({urls[int(m[1])]})', text)
p.write_text(text, encoding='utf-8')
print('Zredagowano własne dokumenty i dodano bezpośrednie odnośniki R1–R24.')
