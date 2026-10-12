---
id: AUD-20261009-VIT-K1000-REZIMY-ASTRA
date: 2026-10-09
status: ROZLICZONY
model: gpt-6-astra
reasoning_effort: xhigh
scope: audyt istniejącego oracle1500, A/B i osobnego DMD/DMDc; bez treningu
poprzednie_audyty:
  - outputs/research_k1000_czasteczki_cel_sztuczny_20261008/RAPORT.md
  - outputs/vit_k1000_1500_pelne_latenty_90_20261008/RAPORT_KONCOWY.md
  - outputs/vit_k1000_taniec_czasteczek_20261009/README_DANYCH.md
artefakty_dowodowe:
  - outputs/vit_k1000_analiza_rezimow_astra_20261009/DIAGNOSTYKA_LOKALNA.json
  - outputs/vit_k1000_analiza_rezimow_astra_20261009/DIAGNOSTYKA_EDEN.json
  - outputs/vit_k1000_analiza_rezimow_astra_20261009/PROVENANCE_ODCZYTU.json
  - outputs/vit_k1000_analiza_rezimow_astra_20261009/SACCT.txt
---

# Dlaczego korekty A/B nie przeniosły sukcesu oracle i co naprawdę sprawdzono temporalnie

**Werdykt:** wykonano rzeczywistą, lecz ograniczoną analizę DMD/DMDc na 128 trajektoriach. Nie użyto jej do zbudowania korekty B. Nie wykonano pełnej analizy ruchu wszystkich 900 000 cząsteczek co krok. A jest średnią końcowych delt, a B statycznym grupowaniem końcowych przemieszczeń z przybliżonym przypisaniem z naturalnego latentu.

**Najmocniejszy nowy wynik audytu:** do tych średnich włączono również 217 nieudanych optymalizacji. Ich delty stanowią **99,564 procent energii wszystkich końcowych delt**. W udanych przypadkach faza minimalizacji normy sprowadziła większość tokenów innych niż cel prawie do punktu wyjścia. W rezultacie średnia A niemal odtwarza wkład przypadków nieudanych, zamiast typowej udanej korekty. To można wykazać ze wszystkich zapisanych norm, bez ponownego treningu.

**Drugie kluczowe ograniczenie:** w panelu DMD/DMDc jest **zero tokenów wybranych jako cel oracle**. Jest 16 obrazów i po osiem stałych pozycji siatki, żadna nie pokrywa się z docelową kotwicą tych obrazów. Test badał więc dynamikę tokenów niebędących celem, głównie ich tłumienie i późniejszy powrót, a nie pełną dynamikę naprawy lokalizacji narzędzia.

Pomysł użytkownika jest sensownym kierunkiem badawczym. Dotychczasowe A/B nie są jego pełnym testem. Nie ma jednak jeszcze dowodu, że wspólne wzorce ruchu dają korektor poprawiający nowe filmy. Najpierw trzeba oddzielić sukcesy od porażek i fazy optymalizacji, a potem sprawdzić, które wzorce można rozpoznać z informacji dostępnych podczas zwykłej inferencji.

## 1. Zakres, wersja wykonania i siła dowodów

Projekt zapisu: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation`. Odczytano instrukcje tego projektu i nadrzędne AGENTS, skill audytu oraz skill eksperymentu. Przed zapisami wykonano `git status --short`; zastany worktree zawierał liczne cudze zmiany. Zapis dotyczy wyłącznie tej rodziny `outputs/` i odpowiadającej jej rodziny `experiments/`. Nie zmieniono rejestru AU w Grant, wspólnego rejestru rodzin ani rodziny Hubble'a. Taki zakres wynika z jawnego ownership użytkownika i ma pierwszeństwo przed standardowym miejscem zapisu skilla.

Analiza używa oznaczeń **POTWIERDZONE**, **PRAWDOPODOBNE**, **BRAK_DANYCH**, **NIEROZSTRZYGAJĄCE**. Fakty z nowych obliczeń to opis zachowanych artefaktów. Nie przedstawiam ich jako ablacji przyczynowej ani dowodu generalizacji.

Źródła historyczne i ich pochodzenie:

- Checkpoint K1000 SHA256: `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`.
- Manifest 1500 obrazów SHA256: `33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7`.
- Historyczny inventory kodu: [CODE_INVENTORY.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/provenance/CODE_INVENTORY.json).
- Zgodność końcowego wdrożenia z lokalnym kodem, z uwzględnieniem CRLF/LF: [FINAL_RUNTIME_CODE_MATCH.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/provenance/FINAL_RUNTIME_CODE_MATCH.json).
- Nowy odczyt kodu i logów Edena, SHA256 oraz porównanie z historycznym inventory: [PROVENANCE_ODCZYTU.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/PROVENANCE_ODCZYTU.json). Kopie małych źródeł znajdują się w `snapshot_eden/` tej rodziny. Same bieżące pliki worktree nie są traktowane jako dowód wykonania starego joba.
- Ponowny odczyt `sacct`: [SACCT.txt](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/SACCT.txt). Job reżimów `1824879` i DMDc `1824880`: COMPLETED, exit `0:0`; odpowiednio 4 min 7 s i 18 s. Dowodem wyniku DMDc jest również log zgodny z zapisanym modelem i raportem.
- Główny raport poprzednika: [RAPORT_KONCOWY.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/RAPORT_KONCOWY.md).

Bieżący commit nadrzędnego repo: `8607e0dc97e0889d9b357ce57515e8e09efeb3c8`. Rodziny eksperymentów były nieśledzone przez Git, dlatego identyfikatorem naukowego wykonania są hashe, inventory, logi i artefakty, nie sam commit. Wstępny `BASE_CODE_PROVENANCE.json` wskazuje wcześniejszy model zależny od timm; końcowa wersja samodzielna ma inny hash i własny dowód równoważności. Nie mieszam tych wersji.

Wybrane hashe bieżącego odczytu Edena **zgodne bajtowo z historycznym inventory**:

| Plik | SHA256 |
|---|---|
| analiza_dmdc.py | `6e7d0f911240313abb1ab5d656a39e022b29a978ce90ceec593bbc8e66d5e897` |
| analiza_regimow.py | `7d8d649b3be7b719294f6613344ff45c5feea809a8eb8067ac21d50f3d72e101` |
| pelne_latenty.py | `aa888560aa4c15091f54b3a5061e85ac575ba3bc01e7df7919fc5b0de65deded` |
| korekta_model.py | `174c343173ccf96d78849c8f6f5620c78a6a258705ebd5aa1a12122b7b6003d5` |
| model_snapshot.py | `1422336d4472db56ce5c89294e3ab8bf0b17ad7502af0876d9c7e564b2473c37` |

Nie wszystkie pliki późniejszego wdrożenia mają pierwotny hash: `pakuj_modele.py` i jego test zmieniono przy poprawce odczytu opcjonalnego timm; `metryki_source.py` różni się CRLF/LF. Dla wykonanej oceny używam jej własnego `EVALUATION_CONFIG.json` i `FINAL_RUNTIME_CODE_MATCH.json`. Te różnice nie dotyczą kodu odkrywania reżimów, aktualizacji oracle ani regresji DMDc. Eksportowane korekty A/B mają odpowiednio SHA256 `58050439532acc743a3b6581f937ba1dc38ad4d3d0322936c39a21ecfd6b70f2` i `781944903dfb55587f53e2b6af15af479f1c5b3022d94115e373067883cf0fc3`.

Model audytu i effort potwierdzono z `turn_context` bieżącej sesji: **gpt-6-astra, xhigh**. Nie jest to deklaracja wywnioskowana z nazwy katalogu.

## 2. Retrospekcja poprzednich diagnoz

Nie znaleziono istniejącego AU dotyczącego tego samego oracle1500/A/B w celowanym przeszukaniu rejestru Grant. Poniższe lokalne identyfikatory należą wyłącznie do tego raportu. Nie nadano nowych globalnych numerów AU.

| Poprzednie twierdzenie lub interpretacja | Rozliczenie | Status dowodowy i stan |
|---|---|---|
| „Nie było DMD/Koopmana, tylko k-means” | Zbyt szerokie i fałszywe: wykonano osobny model przejść DMD/DMDc w NumPy. Nie wykonano temporalnego korektora B. | USTALENIE_OBALONE / USTALENIE_OBALONE, pewność wysoka |
| „B to PCA16 i 32 centroidy z naturalnego wektora” | Niepełne: discovery widziało z0, kierunek oracle delta i jej normę; assignment widział z0, pozycję i kontekst. Są dwa różne zbiory centroidów. | POTWIERDZONE doprecyzowanie, opis rozliczony |
| „Ruch wczesny i późny przeciwny, więc ruchy się znoszą” | Statystyka segmentów jest prawdziwa. Interpretacja jako przyczyna niepowodzenia A jest zbyt mocna: cofanie występuje w fazie min_norm sukcesów, a średnią zdominowały porażki o zgodnych kierunkach. | POTWIERDZONE dane; NIEROZSTRZYGAJĄCE jako pierwotna przyczyna |
| „Poprawa tokenu narzędzia konkuruje z tłumieniem 599 innych” | Strata rzeczywiście wyróżnia jeden GT-token i tłumi pozostałe. Jednak po sukcesie większość pozostałych wraca do z0. Konkurencja w rankingu jest potwierdzona konstrukcyjnie; jej udział w spadku AP wymaga ablacji. | POTWIERDZONE strukturalnie; PRAWDOPODOBNE jako część przyczyny, OTWARTE |
| „Numer patcha nie ma wspólnego znaczenia obiektowego” | Ten sam patch ma stałą pozycję przestrzenną, lecz zmienną rolę względem obiektu. Nie oznacza to braku jakiejkolwiek informacji semantycznej w z0 lub użyteczności priory pozycyjnych. | POTWIERDZONE ograniczenie stałej delty, OTWARTE |
| „74,647 procent energii to nullspace” | Liczba dotyczy historycznego J128. Nie jest statystyką oracle1500. Nowe wyniki dla panelu podano osobno; nie ekstrapoluję ich na 1500. | USTALENIE_OBALONE tylko dla przeniesienia tej liczby; stary wynik zachowuje swój zakres |
| „Mamy pełny taniec 900 000 cząsteczek co krok” | Pełne stany 600×768 są tylko dla 0/100/500 oraz indywidualnego after. Co krok są 128 trajektorii. | USTALENIE_OBALONE / USTALENIE_OBALONE |
| „After jest stanem 500” | Kontrakt tego nie gwarantuje. Faktycznie wszystkie 1283 sukcesy wybrały 500; 203 z 217 porażek wybrały wcześniejszy krok. | POTWIERDZONE doprecyzowanie |
| „Oracle jest górną granicą możliwości ogona” | Jest dostępnym odniesieniem z GT i dowodem osiągalności znalezionych wyników. Budżetowy optymalizator nie wyznacza udowodnionego matematycznego maksimum; 217 porażek nie dowodzi niemożliwości. | POTWIERDZONE ograniczenie interpretacji |

Retrospekcja J128 opiera się na [wcześniejszym RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/RAPORT.md:11). Tam sam autor wyraźnie ogranicza wynik do historycznego celu. Nie przypisuję temu raportowi twierdzenia, że liczba musi obowiązywać nowe oracle1500.

## 3. Co rzeczywiście zrobiono w A, B i DMDc

### 3.1. Miejsce interwencji i rzeczywisty downstream

To **wyjście GELU w ukrytej warstwie MLP ostatniego bloku transformera**, `bloki[-1].mlp.act`, kształt 600×768. Nie jest to końcowy latent 192D po całym enkoderze. Dalsza ścieżka obejmuje `fc2: 768→192`, gałąź residual ostatniego bloku, końcowy LayerNorm i głowę `Linear(192,192) → GELU → Linear(192,5)`.

Dokładny wzór realizowany przez fast tail, przy stałym obrazie i residual:

\[
q_{i,j}(z)=q^0_{i,j}+L_2W_2(z_{i,j}-z^0_{i,j}),\quad
o_{i,j}=H(\operatorname{LN}(q_{i,j})).
\]

Nie ma już attention mieszającego tokeny za interwencją. Sprzężenie podczas optymalizacji pochodzi m.in. z rankingu logitów i decyzji o fazie, a nie z kolejnego mieszania reprezentacji przez transformer. Token naturalny zawiera kontekst uzyskany wcześniej, ale końcówka działa token po tokenie. Cache `q0` jest zależny od obrazu, więc samo położenie z w 768D nie określa całej odpowiedzi głowy dla różnych obrazów.

Dowody: [pelne_latenty.py:30](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/pelne_latenty.py:30), [fast_tail:58](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/pelne_latenty.py:58), [model_snapshot.py:119](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/model_snapshot.py:119). Reload po zapisie porównywano z rzeczywistym hook-forward, nie tylko z własnym fast tail: [uruchom_oracle.py:279](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/uruchom_oracle.py:279).

Model działał w eval, z zamrożonymi parametrami. Aktywacje optymalizował **nowy Adam per obraz**, nie AdamW odziedziczony po treningu K1000. Zachowany checkpoint AdamW wag jest elementem provenance, ale nie stanem optymalizatora aktywacji: [optimize_latent:275](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/pelne_latenty.py:275).

### 3.2. Oracle i pole after

GT wybiera docelową kotwicę oraz docelową ramkę. Faza fit optymalizuje confidence celu, tłumienie pozostałych logitów, ranking i ramkę. Po **pierwszym** osiągnięciu obu progów następuje trwałe przejście do min_norm, również jeśli sukces chwilowo zniknie. Nowa strata minimalizuje ruch z karami za naruszenie marginesów confidence/rankingu/IoU. Nie jest to jeden stacjonarny cel przez 500 kroków.

Learning rate: kroki 1–100: 0,05; 101–250: 0,15; 251–400: 0,05; 401–500: 0,01. Po każdej aktualizacji jest clamp do `-0.16997121`. `after` wybiera odwiedzony sukces o najmniejszym RMS ruchu; gdy sukcesu nie ma, wybiera najlepszy odwiedzony stan według leksykograficznego kryterium jakości. **After porażki nie jest rozwiązaniem problemu minimalnej normy z osiągniętymi progami.**

Dowody: [straty:193](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/pelne_latenty.py:193), [wybór stanu i pętla:311](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/pelne_latenty.py:311).

Uwaga do telemetrii: `phase_min_norm[t]` informuje, że po obejrzeniu stanu t sukces jest już znany; określa cel następnego przejścia t→t+1. `gradient_rms[t]` dla t>0 pochodzi z aktualizacji t−1→t. Pole `fit_objective` nadal zapisuje starą stratę fit także w min_norm; nie jest wtedy wartością aktualnie minimalizowanej straty. To trzeba uwzględnić przy korelacjach i wyrównywaniu zdarzeń.

### 3.3. Metoda A

\[
\Delta^A_j=\frac1N\sum_i(z^{after}_{i,j}-z^0_{i,j}),\qquad
z'_{i,j}=\max(z^0_{i,j}+\Delta^A_j,\mathrm{floor}).
\]

Średnia liczona po obrazach osobno dla 600 patchy. Brak optymalizacji wspólnej delty przez stratę detekcji, brak uczenia przejść czasowych, brak selekcji wyłącznie sukcesów. Wszystkie 1500 `artifact['delta']` trafia do wejścia, także porażki. Wagi delta nie są wyrównywane według normy.

Dowody: [analiza_regimow.py:155](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/analiza_regimow.py:155), [wczytanie delt:269](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/analiza_regimow.py:269), [MeanCorrection](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/korekta_model.py:8).

### 3.4. Metoda B: discovery i assignment to różne problemy

1. PCA16 osobno dla naturalnego z0 oraz **jednostkowych kierunków końcowej delty**. Próbka PCA: 50 000 par obraz–token, seed 20261008. Bez przebiegu co krok.
2. Discovery: 33 cechy, czyli 16 współrzędnych PCA z0, 16 współrzędnych PCA kierunku delty oraz `log1p(norm(delta))`. Każda cecha jest standaryzowana. K-means: 32 reżimy, próbka 100 000 par, do 25 iteracji.
3. Assignment: 34 cechy, czyli 16 współrzędnych PCA z0, 2 współrzędne pozycji tokenu i 16 średnich współrzędnych kontekstu obrazu. Centroid każdej etykiety discovery liczony w tej nowej przestrzeni. Inferencja wybiera najbliższy centroid assignment.
4. Korekta to tablica `32×600×768`: średnia delta dla reżimu i patcha, z shrinkage 20 do A:

\[
\Delta^B_{r,j}=\frac{\sum_{i:r_{ij}=r}\Delta^{oracle}_{i,j}+20\Delta^A_j}{n_{r,j}+20}.
\]

5. Inferencja dodaje wybraną deltę do własnego naturalnego z0 i stosuje clamp. Nie pobiera GT, przyszłej delty, DMD_MODEL ani DMDC_MODEL.

To **statyczne warunkowanie końcowej korekty**, nie identyfikacja temporalnych reżimów ruchu. Sama nazwa „reżim” nie oznacza czasów przebywania, przełączeń, widma operatora ani zależności następnego kroku od poprzedniego.

Dowody: [fit_regime_model:92](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/analiza_regimow.py:92), [assignment:142](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/analiza_regimow.py:142), [RegimeCorrection:46](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/korekta_model.py:46).

### 3.5. DMD/DMDc: deklaracja, przygotowany kod, wykonanie

| Warstwa dowodu | Co ustalono |
|---|---|
| Dokument metodologii | Wprost zapowiada regresję NumPy i oddzielenie DMDc od transferu. Linki do deeptime/PyDMD są źródłami metody, nie deklaracją importu. |
| Kod | `analiza_dmdc.py` używa NumPy, własnego `_ridge`, PCA z `analiza_regimow` i readera artefaktów. Brak importów deeptime/PyDMD. |
| Wywołanie | Slurm `dmdc.slurm`: `analiza_dmdc.py --production ... --control-panel ... --output ... --rank 32`. |
| Wykonanie | Job 1824880 COMPLETED; log zawiera 128 trajektorii, 64 000 przejść i metryki zgodne z DMDC_REPORT. Zachowano DMDC_MODEL.npz. |
| Transfer | `not_used_for_transfer_rule: true`; eksport A/B czyta modele regimes. Evaluation zależy od joba regimes; DMDc jest potrzebny dopiero do finalizacji raportu. Brak ścieżki danych DMDc→korekta B. |

DMD jest tutaj modelem afinicznym `y=A x+b` w PCA32 stanu; DMDc dodaje `B u`, gdzie u to PCA32 konkatenacji **gradientu, m1 i m2** (łącznie 2304 surowe współrzędne). To zredukowany predyktor typu DMD/DMDc z ridge, nie pełna analiza modalna: skrypt nie wyznacza ani nie interpretuje eigenvalues/modes, nie grupuje trajektorii na podstawie dynamiki i nie wykonuje EDMD z nieliniowym słownikiem.

Parowanie `z[:, :-1]` z `z[:, 1:]` wykonano **przed** spłaszczeniem. Nie ma przejść między kolejnymi obrazami ani tokenami. Gradient i pamięć zapisano przed `optimizer.step`; przyszła delta nie jest sterowaniem. PCA stanu i sterowania dopasowano tylko na trajektoriach uczących. Holdout: obrazy 0, 500, 999, 1499, 32 trajektorie; train: 12 obrazów, 96 trajektorii. To 48 000 par train i 16 000 test.

| Predyktor | RMSE PCA32, test | R² stanu, test |
|---|---:|---:|
| Persistence, następny stan = obecny | 0,080696899 | 0,999844624 |
| DMD | 0,166732089 | 0,999336700 |
| DMDc | 0,166738362 | 0,999336651 |

DMDc ma około 2,07 razy większy RMSE od persistence. R² bliskie 1 nie oznacza dobrej predykcji ruchu: bezruch wygrywa, gdy zmiana jest mała wobec różnic położenia. Miara jest w PCA32, nie pełnym 768D i nie mAP. Nie dowodzi to, że wszystkie metody Koopmana zawodzą, ani że brak przewidywalnej struktury.

Dowody: [analiza_dmdc.py:43–120](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/analiza_dmdc.py:43), [DMDC_REPORT.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/eden_results/dmdc/DMDC_REPORT.json), [submit_production.sh:35](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/eden/submit_production.sh:35), [pakuj_modele.py](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/pakuj_modele.py), [METHODOLOGY.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_k1000_1500_pelne_latenty_90_20261008/METHODOLOGY.md:17).

**Odpowiedź na pytanie o biblioteki:** w tym wykonaniu A/B i diagnostyki DMDc nie użyto deeptime ani PyDMD. Wykonano własną implementację NumPy. Nie oznacza to „braku DMD”; biblioteka i metoda to odrębne fakty. Nie rozszerzam tej odpowiedzi na wszystkie starsze rodziny Koopmana obecne w repozytorium.

## 4. Wyniki detekcji i granice ich porównania

| Metoda | AP50 | AP75 | mAP50:95 | confidence≥0,9 oraz IoU≥0,9 |
|---|---:|---:|---:|---:|
| Naturalny K1000 | 0,626887827 | 0,131096931 | 0,245496238 | 0/1500 |
| A, in-sample | 0,631022894 | 0,115979549 | 0,239411249 | 0/1500 |
| B, in-sample | 0,622991980 | 0,131139018 | 0,244481144 | 0/1500 |
| A, holdout po filmach | 0,630687565 | 0,116778068 | 0,239095820 | 0/1500 |
| B, holdout po filmach | 0,628031663 | 0,127554300 | 0,242896830 | 0/1500 |
| Oracle per obraz, z GT | 0,975087294 | 0,940327757 | 0,861131039 | 1283/1500 |

To nie jest wynik „A niczego nie poprawiła”: AP50 nieco wzrosło. Jednocześnie spadły AP75 i mAP, a top-1 confidence spadło dla 1481 obrazów w A i 1490 w B. Poprawa obu wskaźników Pareto dotyczyła odpowiednio 4 i 3 obrazów. Nie podano niepewności bootstrapowej tych różnic; raportujemy zaobserwowane wartości, nie istotność statystyczną.

Definicja AP: jedna klasa, top-10 lokalnych maksimów, globalny ranking score, dopasowanie jeden-do-jednego per obraz, 101-punktowa interpolacja; mAP po progach IoU 0,50:0,05:0,95. Padding dekodera ze score≤0 wyłączono. Confidence/IoU w parze progów opisują rzeczywistą detekcję top-1. Nie należy utożsamiać obu90 z AP90 ani z confidence celu, gdy cel nie jest top-1.

Wszystkie warianty porównano na tym samym poprawionym GT po przeskalowaniu osi do 480×320. Historyczne ramki kampanii różniły się dla 203 obrazów; nie były używane jako GT tych metryk. Foldy uczą od nowa całe PCA/centroidy/tablice bez filmów danego foldu. To prawidłowa wewnętrzna kontrola transferu korektora, ale nie niezależny zewnętrzny test generalizacji całego K1000.

Źródła: [EVALUATION_SUMMARY.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/eden_results/evaluation/EVALUATION_SUMMARY.json), [EVALUATION_CONFIG.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/eden_results/evaluation/EVALUATION_CONFIG.json), [ewaluuj_korekty.py](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/ewaluuj_korekty.py), [MANIFEST_CORRECTION_AUDIT.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/provenance/MANIFEST_CORRECTION_AUDIT.json).

## 5. Mechanika niepowodzenia: co jest potwierdzone, a co pozostaje hipotezą

<a id="AUD-20261009-VIT-K1000-REZIMY-ASTRA-A-01"></a>
### A-01. Zmieszano udane korekty minimalnej normy z nieudanymi dużymi ruchami

**P1, POTWIERDZONE, pewność wysoka, OTWARTE.** Kod nie filtruje porażek, a ich skala dominuje końcowy zbiór. Z nowych obliczeń na wszystkich zapisanych normach 768D:

| Grupa | Obrazy | Udział energii Σ‖delta‖² | Mediana normy tokenów innych niż cel |
|---|---:|---:|---:|
| Sukces | 1283 | 0,435920 procent | 0,000001420 |
| Wyczerpanie budżetu | 217 | 99,564080 procent | 95,833794 |

W sukcesach docelowy token zachowuje medianę ruchu 29,450281. To zupełnie inny obiekt statystyczny niż duża delta niemal każdego tokenu w porażce. „Wspólna średnia ruchu optymalizatora” łączy tu małą udaną korektę zadaniową z wielką nieudaną interwencją.

Można wzmocnić wniosek bez dostępu do wszystkich surowych wektorów. Dla wkładu sukcesów do A zachodzi nierówność trójkąta:

\[
S_j=\frac1{1500}\sum_{i\in sukces}\Delta_{ij},\quad
F_j=\frac1{1500}\sum_{i\in porażka}\Delta_{ij},\quad A_j=S_j+F_j,
\]
\[
\frac{\|A_j-F_j\|}{\|A_j\|}\le
\frac{\sum_{i\in sukces}\|\Delta_{ij}\|}{1500\|A_j\|}.
\]

Prawa strona ma medianę **0,008136**, maksimum **0,086767** po 600 patchach. Zatem wkład sukcesów może zmienić wektor A względem wkładu porażek o najwyżej 8,68 procent jego normy na dowolnym patchu, a na połowie patchy o najwyżej około 0,81 procent. `F` ma mianownik 1500; nie mylić go ze średnią samych 217 porażek.

To jest dowód dominacji porażek w wektorze A, a nie tylko w energii. Natomiast twierdzenie „usunięcie porażek na pewno poprawi mAP” byłoby przedwczesne. Do niego potrzebna jest ocena ablacją.

Średnia A ma medianę normy 10,717434 na patch. Stosunek `‖mean delta‖ / mean ‖delta‖` ma medianę **0,900308**. To **nie** wygląda na masowe kasowanie przeciwstawnych końcowych kierunków pomiędzy obrazami. Energia projekcji na stałą deltę to tylko 9,434238 procent całkowitej energii, ale wynika to także z wielkiej heterogeniczności magnitud i udziału prawie zerowych ruchów. Niski udział energii sam w sobie nie dowodzi konfliktu kierunków.

Dowód: [DIAGNOSTYKA_LOKALNA.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/DIAGNOSTYKA_LOKALNA.json), pola `endpoint_phase_split`, `success_contribution_to_A_norm_upper_bound_per_patch`, `mean_delta`. Obliczenia wykorzystują istniejące `MOVEMENT_ARRAYS.npz`, `MEAN_DELTA_ALL.npz` i metadane wizualizacji, bez użycia współrzędnych PCA3 do wniosków mechanistycznych.

<a id="AUD-20261009-VIT-K1000-REZIMY-ASTRA-A-02"></a>
### A-02. Znaczną część grupowania kierunków B tworzy niemal zerowy ruch

**P1, POTWIERDZONE zjawisko; PRAWDOPODOBNE pogorszenie discovery, OTWARTE.** Dla 719 065 z 900 000 delt norma jest mniejsza niż 0,0001. W odtworzonej indeksowej próbce PCA kierunków B odsetek wynosi **80,112 procent**. Kod dzieli deltę przez `max(norm, 1e-12)`, więc kierunki takich małych reszt dostają praktycznie normę jednostkową tak jak kierunki istotnego ruchu. PCA kierunku i standaryzacja jego cech nie odróżniają automatycznie ruchu znaczącego od śladu powrotu do z0. Nie twierdzę, że każda mała reszta jest wyłącznie błędem zaokrąglenia; jej semantyczna istotność nie została wykazana.

Trafność przypisania etykiety discovery z cech naturalnych wynosi 36,057444 procent na train, balanced recall 35,362081 procent. To jest wyraźnie **powyżej majority 4,647333 procent**, a więc nie wolno mówić „reżimy są całkowicie nieprzewidywalne”. Jednocześnie około 64 procent tokenów otrzymuje inną etykietę niż oracle discovery. Trafność etykiet nie mierzy jeszcze jakości korekty; podobne etykiety mogą mieć różne delty i odwrotnie.

Tablica reżim×patch ma medianę 24 obserwacje na komórkę. 1929 z 19 200 komórek jest pustych, a 8802 ma mniej niż 20 obserwacji. Shrinkage 20 istotnie przyciąga wiele komórek do A, która jest zdominowana przez porażki. To drugi kanał przeniesienia tego problemu do B. Dopiero kontrola z prawdziwą etykietą discovery na odłożonych obrazach odróżni złą tablicę ruchów od złego przypisania; taka kontrola byłaby oracle diagnostyką, nie wdrażalnym wariantem.

Dowody: [konstrukcja kierunków:116](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/analiza_regimow.py:116), [REGIME_DIAGNOSTICS.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/eden_results/regimes/REGIME_DIAGNOSTICS.json), `REGIME_MODEL_ALL.npz/regime_token_counts` i własna diagnostyka lokalna.

<a id="AUD-20261009-VIT-K1000-REZIMY-ASTRA-A-03"></a>
### A-03. Odwracanie segmentów opisuje przede wszystkim zmianę celu optymalizacji

**P1, POTWIERDZONE, pewność wysoka; wcześniejsza przyczynowa interpretacja NIEROZSTRZYGAJĄCA.** Historyczne 768 954 przeciwne kierunki i mediana cos≈−1 są zgodne z zapisem. Po rozdziale według pierwszego sukcesu:

- Sukces do kroku 100: 931 obrazów, przeciwne segmenty 0→100 i 100→500 dla 99,9121 procent tokenów.
- Sukces po kroku 100: 352 obrazy, przeciwne segmenty dla 99,8319 procent tokenów.
- Brak sukcesu: 217 obrazów, przeciwne segmenty dla **0 procent** tokenów.

W naturalny sposób wyjaśnia to fit, następnie powrót ku z0 w min_norm. Średnia po sztywnym t łączy obrazy w różnych fazach. W kroku 100 aż 931 obrazów osiągnęło sukces kiedykolwiek, ale oba progi aktualnie spełniało tylko 863. Przełączenie fazy jest zdarzeniem historycznym, nie równoznacznym z bieżącym sukcesem. W kroku 500 wszystkie 1283 kiedyś udane obrazy znów spełniały progi.

Cosinus dwóch długich segmentów nie pokazuje, ile razy kierunek zmienił się **wewnątrz** segmentu. Nie jest dowodem oscylacji, metastabilności ani jednego wspólnego mechanizmu wszystkich tokenów. Filtr minimalnej normy obu segmentów zachowuje ogólny wynik, więc nie jest on wyłącznie skutkiem dzielenia przez niemal zero; interpretacja faz nadal jest konieczna.

Dowody: [analiza_ruchu.py:60](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/analiza_ruchu.py:60), [DIAGNOSTYKA_LOKALNA.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/DIAGNOSTYKA_LOKALNA.json), pola `reversal_by_phase_group` i `reversal_by_minimum_segment_norm`.

<a id="AUD-20261009-VIT-K1000-REZIMY-ASTRA-A-04"></a>
### A-04. Stały patch nie określa zadania i nie zna potrzebnej ramki

**P1, POTWIERDZONE ograniczenie; przyczynowy udział w spadku AP PRAWDOPODOBNY, OTWARTE.** Docelowe kotwice występują na 285 z 600 pozycji. Najczęstsza pozycja jest celem tylko dla 24/1500 obrazów, czyli 1,6 procent. Ten sam patch zwykle nie jest celem, nawet jeżeli bywa nim w innych obrazach. „Pozostałe 599” nie oznacza automatycznie czystego tła: niektóre mogą przecinać narzędzie, ale strata tej procedury traktuje je jako niecelowe konkurencyjne detekcje.

A zawsze dodaje ten sam wektor w danym patchu, lecz nieliniowy ogon może reagować na niego odmiennie zależnie od z0 i residual. Stała delta nie jest matematycznie skazana na porażkę. Nie daje jednak jawnego mechanizmu wyboru celu, wymaganej zmiany rozmiaru ramki, kierunku przesunięcia ani warunkowania na stan całego obrazu. B częściowo warunkuje korektę, ale średni kontekst jest ubogi i usuwa układ przestrzenny pozostałych tokenów.

Strata fit `softplus(logit_non_target+2).mean()` i cross-entropy rankingu rzeczywiście kieruje niecelowe logity w dół. Spadek confidence dla 98,73 procent obrazów w A i 99,33 procent w B jest z tym zgodny. Brakuje jednak ablacji: tylko docelowe patche, tylko konkurenci, sukces-only, równoważenie norm, skala delty. Bez niej nie da się przypisać całego spadku AP jednemu składnikowi straty. Dane nie uzasadniają tezy, że „599 ma zawsze 599 razy większy gradient” — strata po negatywach jest uśredniona, a Adam dodatkowo skaluje współrzędne.

### A-05. Nullspace, osiągalność i GT: trzy odrębne kwestie

<a id="AUD-20261009-VIT-K1000-REZIMY-ASTRA-A-05"></a>
**P1, POTWIERDZONE ograniczenia strukturalne; rozmiar dla całego zbioru BRAK_DANYCH.** `fc2` ma kształt 192×768. Dla jego rangi 192 istnieje 576-wymiarowy kernel. Czysta delta w tym kernelu nie zmienia końcówki przy stałym residual. To nullspace **fc2**, bezpieczna podprzestrzeń niewidoczna dla całego ogona; nie jest to pełny, globalny nullspace nieliniowej głowy. Lokalny Jacobian głowy może mieć dodatkowe kierunki niewidoczne pierwszorzędowo, ale nie muszą być niewidoczne przy skończonej interwencji.

Gradient czystej straty detekcji przechodzi przez W2 i leży w jego rowspace. Aktualizacja Adam po skalowaniu współrzędnych, komponent normy oraz clamp nie muszą zachować tej podprzestrzeni. Dlatego nie należy dopasowywać korektora wyłącznie MSE surowych 768 współrzędnych i zakładać, że cały błąd jest zadaniowo ważny. Po odrzuceniu nullspace trzeba ponownie uwzględnić clamp: usunięcie komponentu null i późniejsza projekcja na floor może zmienić efekt.

Granica GELU od dołu jest tylko koniecznym warunkiem naturalności. `norm2/fc1` i wspólna reprezentacja 192D narzucają sprzężenia pomiędzy 768 współrzędnymi; sam clamp ich nie spełnia. Wcześniejszy audyt udowodnił nieosiągalność **J128**, nie wszystkich nowych endpointów. Dla oracle1500 nie wykonano tu pełnego audytu naturalnego manifold ani inwersji obrazów. To nie przekreśla korektora dodawanego bezpośrednio do aktywacji: taki adapter musi generalizować i poprawiać detekcję, niekoniecznie odtwarzać dokładny naturalny latent jakiegoś innego obrazu.

Dodatkowo 32/1500 ramek ma geometryczny górny pułap IoU kotwic poniżej 0,9 według istniejącej geometrii dekodera; wszystkie 32 są w grupie 217 porażek. To osobne ograniczenie siatki kotwic, nie dowód o manifoldzie. Pozostałych 185 porażek nie wolno uznać za geometrycznie niemożliwe na tej podstawie. Budżet, cel i parametry optymalizacji nie są dowodem nieosiągalności.

GT w teacherze oracle jest dozwolonym nadzorem, jeśli discovery, dobór hiperparametrów i uczenie korektora są zamknięte w train. Nie jest dozwolonym wejściem inference. DMDc używa gradientu celu zależnego od GT; nawet idealne odtwarzanie tych trajektorii nie daje automatycznie regulatora bez GT. Ocena rollout z podawanym prawdziwym gradientem testowym jest diagnostyką dynamiki, nie uczciwą oceną wdrażalnego correctora.

<a id="AUD-20261009-VIT-K1000-REZIMY-ASTRA-A-06"></a>
## 6. A-06. Jakie dane temporalne naprawdę istnieją

**P1, POTWIERDZONE braki pokrycia, pewność wysoka, OTWARTE.**

| Zakres | Faktyczny zapis | Co pozwala ustalić |
|---|---|---|
| Wszystkie 1500×600 tokenów | before 0, trajectory 100/500, after, delta=after−before, pełne 768D | Endpointy, dwa długie segmenty, normy i geometria pełnej przestrzeni; nie każdy krok |
| 16 obrazów×8 tokenów | control_z: 501×8×768, gradient/m1/m2: po 500×8×768 | Rzeczywiste 64 000 przejść, lokalne zwroty i pamięć optymalizatora wybranego panelu |
| Wszystkie 1500 obrazów | telemetria 501 kroków: confidence, IoU, delta_rms/l2, gradient_rms, phase, top_token, success, fit_objective, lr | Zdarzenia, fazy, efekty na detekcji, ruch zagregowany po obrazie; nie kierunki wszystkich tokenów |
| Lokalna wizualizacja | PCA3 wszystkich punktów 0/100/500 i 128 gęstych trajektorii, normy liczone w 768D | Wspólny podgląd; nie rekonstrukcja 768D i nie pełny timeline 900 000 |

Panel ma stałe tokeny `[0,29,145,284,315,454,570,599]` dla indeksów obrazów `[0,100,200,300,400,500,600,700,799,899,999,1099,1199,1299,1399,1499]`. **Żaden z 16 target_token nie należy do tych ośmiu pozycji swojego obrazu.** Liczba 128 nie oznacza 128 obrazów ani próby zbalansowanej na tokeny naprawiające narzędzie.

Ponadto jedyna porażka panelu, obraz 0, znalazła się w holdoucie DMDc. Train zawiera tylko 12 sukcesów. Test zawiera trzy sukcesy i jedną porażkę. To duża zmiana udziału faz i długości fit pomiędzy train/test, mimo poprawnego nierozcinania trajektorii. Wszystkie 16 obrazów panelu pochodzi z różnych filmów, więc w tej próbce nie ma wspólnego filmu między train i test; mała liczba filmów i brak celów nadal mocno ograniczają wynik.

Wspólna PCA3 wizualizacji ma 79,288941 procent wariancji **próbki fit** 64×600×3 stanów, bez standaryzacji kanałów, ze wspólną bazą float64. Nie jest to 79 procent zachowanej informacji o detekcji ani o prędkościach. Użyte w tym audycie globalne normy i cosinusy pochodzą z oryginalnych 768D; projekcja obrazu nie jest dowodem mechanistycznym.

Dowody: [przygotuj_manifest.py:101](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/snapshot_eden/code/przygotuj_manifest.py:101), [CONTROL_PANEL.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_1500_pelne_latenty_90_20261008/provenance/CONTROL_PANEL.json), [README_DANYCH.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_taniec_czasteczek_20261009/README_DANYCH.md), [VALIDATION.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_taniec_czasteczek_20261009/data/VALIDATION.json), [LOCAL_TRANSFER.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_taniec_czasteczek_20261009/data/LOCAL_TRANSFER.json).

Nie da się odzyskać brakujących zwrotów z trzech klatek ani z norm telemetrycznych. Uzupełnienie ich interpolacją stworzyłoby przyjęty model ruchu, a nie odczyt rzeczywistej trajektorii. Pełną dynamikę wszystkich tokenów należałoby zarejestrować w osobno autoryzowanym ponowieniu.

## 7. Nowy, tani odczyt pełnych 128 trajektorii i istniejącego operatora

**Wykonano**, bez fit, nowych jobów ani inference obrazów: odczyt 16 plików panelu na jednym wątku CPU, około 123 s obliczeń/odczytu. Zdalny skrypt niczego nie zapisuje, drukuje mały raport JSON; zapis odbył się lokalnie. To nie jest pełna nowa analiza 1500 przebiegów. [DIAGNOSTYKA_EDEN.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/DIAGNOSTYKA_EDEN.json) zawiera hashe wszystkich 16 NPZ, kształty tablic i wyniki per obraz.

### 7.1. Oś czasu i pamięć optymalizatora są spójne

Ze stanu z_t, gradientu g_t i chwil sprzed kroku odtworzono analitycznie:

\[
m^+=0.9m+0.1g_t,\quad v^+=0.999v+0.001g_t^2,
\]
\[
z_{t+1}=\max\left(z_t-\eta_{t+1}
\frac{m^+/(1-0.9^{t+1})}{\sqrt{v^+/(1-0.999^{t+1})}+10^{-8}},\mathrm{floor}\right).
\]

Maksymalny błąd współrzędnej wobec zapisu wyniósł **9,50181×10⁻⁷**. To mocny dowód poprawnego wyrównania zapisów g/m/v i z dla tych trajektorii. Nie dowodzi dostępności gradientu bez GT ani kompletności stanu całego obrazu. Replay zna rzeczywisty gradient; jest kontrolą aparatu, nie samodzielnym korektorem.

### 7.2. Zwroty co krok są skoncentrowane wokół zmiany fazy

Cosinus kolejnych przyrostów obliczono w 768D, tylko gdy obie normy kroku przekraczały 0,0001. Po rozdzieleniu przejść wewnątrz fit, wewnątrz min_norm i przekraczających zmianę fazy:

- Na przełączeniu fazy było 120 par przyrostów, z czego **118 miało ujemny cosinus**.
- Wewnątrz fit nie było ujemnego cosinusa w zakwalifikowanych parach panelu.
- Wewnątrz min_norm wystąpiło 86 takich par; większość kierunków pozostawała dodatnio skorelowana. Mediany per obraz podano w JSON, nie traktowano ich jako mediany połączonej próby.

To bezpośrednio wspiera interpretację „zmiana zadania i powrót”, a nie nieustanny nieuporządkowany taniec wszystkich cząsteczek. Próg odcina ruchy zbyt małe do stabilnej interpretacji kąta; nie wolno z tej próby wnioskować o brakujących trajektoriach docelowych tokenów.

### 7.3. DMD przegrywa w fit, ale wygrywa w min_norm

Ponownie zastosowano **zapisane** współczynniki DMD/DMDc, bez ponownego dopasowania. Odtworzone zbiorcze RMSE zgadza się z historycznym raportem do około 5×10⁻⁸. Rozdział testu po fazie źródłowego stanu:

| Faza testu | Liczba przejść | Persistence RMSE | DMD RMSE | DMDc RMSE |
|---|---:|---:|---:|---:|
| Wszystkie | 16 000 | 0,080696897 | 0,166732117 | 0,166738409 |
| Fit | 5616 | 0,079248059 | 0,263767139 | 0,263777709 |
| Min_norm | 10 384 | 0,081469740 | 0,072160577 | 0,072162080 |

W min_norm DMD zmniejsza RMSE względem persistence o około **11,43 procent**. W fit wyraźnie przegrywa; 4000 z 5616 przejść fit testu należy do jedynej porażki panelu, której rodzaju nie ma w train. To nie oznacza automatycznie, że cały błąd pochodzi z tego obrazu, ale pokazuje istotny shift struktury testu. Wniosek „DMD niczego temporalnego nie uchwycił” byłby nieprawdziwy.

<a id="AUD-20261009-VIT-K1000-REZIMY-ASTRA-A-07"></a>
### 7.4. A-07. Skala wejść utrudniała DMDc użycie pamięci

**P1, POTWIERDZONE niezgodne skale i regularyzacja; wielkość poprawy po zmianie BRAK_DANYCH, OTWARTE.**

Na train panelu RMS bloków sterowania wynosi:

| Blok | RMS |
|---|---:|
| Gradient | 1,905816×10⁻⁶ |
| m1 | 1,494757×10⁻⁶ |
| m2 | 1,554578×10⁻¹¹ |

PCA konkatenacji tych bloków była centrowana, lecz nie standaryzowana blokowo. Surowy m2 jest o około pięć rzędów mniejszy od g/m1. Jego adaptacyjny wpływ w Adam wynika z pierwiastka, dzielenia, epsilon i bias correction, nie z liniowej wielkości m2. W tym predyktorze nie ma też jawnego lr, czasu ani fazy.

Dodatkowy problem to `_ridge`: jedna kara `lambda=1e-5*trace(FᵀF)/65` dla 32 cech stanu, 32 sterowania i interceptu, w surowych jednostkach. Już kolumna jedynek daje **lambda≥0,0073846**, podczas gdy całkowita suma kwadratów wycentrowanych wszystkich 2304 cech sterowania train wynosi tylko **0,0002142993**. Projekcja ortogonalna do PCA32 nie może jej zwiększyć. Zatem lambda jest co najmniej **34,46 razy większa od całej energii macierzy sterowania**, a wkład stanu zwiększa tę dysproporcję jeszcze bardziej. To potwierdzona niezgodność skali regularyzacji między blokami. Nie liczę z tego automatycznie wielkości poprawy po zmianie skalowania: to wymaga osobnego fit i holdoutu.

Mała różnica DMD i DMDc **nie dowodzi braku użyteczności gradientu lub pamięci**. Właśnie jawny wzór Adama z tymi wejściami niemal dokładnie odtwarza krok. Wykonana regresja jest słabym testem znaczenia tych informacji, choć pozostaje ważnym negatywnym wynikiem tej konkretnej konfiguracji.

### 7.5. Nowy wynik nullspace dotyczy tylko panelu

Z rzeczywistego checkpointa odczytano `bloki.5.mlp.fc2.weight`, SVD w float64 potwierdza rangę 192 i nullity 576. Dla endpointów **wszystkich 600 tokenów każdego z 16 obrazów panelu**, nie tylko ośmiu trajektorii:

- 15 sukcesów: 47,696593 procent energii after−before w kernelu fc2, suma energii 32 383,48.
- Jedna porażka, obraz 0: 54,154771 procent, suma energii 7 078 286,66.

To proporcje sum energii grup, nie średnia procentów po obrazach. Nie jest to 74,647 procent J128 i nie jest to estymata reprezentatywna dla wszystkich 1500 obrazów. Dowodzi natomiast, że również nowe endpointy mają istotną składową niewidoczną dla fc2. Nie wykonano ablacji mAP po rowspace-only/nullspace-only; jej wyniku nie wolno dopowiadać.

## 8. Poprawny projekt analizy temporalnej

Poniższy projekt **nie został uruchomiony**. Jego celem jest odróżnienie wspólnego wzorca przydatnego dla detekcji od regularności narzuconej przez Adam, schedule, clamp i przełączenie celu.

### 8.1. Co nazywamy stanem i czasem

Cząsteczka to wektor jednego tokenu w jednej wspólnej przestrzeni 768D. Makrowektory są wspólną bazą e_j; mikrowektory cząsteczki to jej składowe z_j e_j. Globalnego układu nie zastępuje średnia tokenów. Każdy obraz ma osobną trajektorię, a t oznacza numer sztucznej aktualizacji aktywacji przy **stałych wagach**, nie klatkę filmu i nie krok uczenia K1000.

Naturalny stan wszystkich tokenów oznaczmy Z0_i, aktualny Z_i,t. Trzeba zachować identyfikator obrazu, filmu, tokenu, krok, fazę i wersję operatora. Dla modelu jednego tokenu potrzebny może być kontekst innych tokenów, ponieważ ranking i zmiana fazy zależą od całego obrazu.

Pełny stan wyjaśniający aktualizację może mieć postać:

\[
s_{i,t}=(Z_{i,t},M_{i,t},V_{i,t},Z^0_i,q^0_i,c_i,t,\eta_t,\phi_{i,t}),
\]

gdzie c_i zawiera warunek obrazu/celu dostępny **w badaniu teacherowym**, a phi oznacza aktywny cel optymalizacji. Jeżeli g_t traktujemy jako wejście, nie musi być składnikiem stanu. Jeżeli chcemy przewidywać g_t, potrzebujemy warunku GT lub innego jawnie zdefiniowanego celu. Z samego z_t nie wynika stan pamięci Adama ani to, czy sukces nastąpił w przeszłości.

### 8.2. Wspólna geometria, redukcja i przyrosty

1. Zostawić 768D jako punkt odniesienia i raz ustalić bazę z train. Nie dopasowywać osi osobno do obrazu ani kroku. Jeśli standaryzować kanały/bloki, zapisać jedną transformację fit tylko na train. Standardyzacja zmienia metrykę: raportować też normy i kąty w pierwotnej przestrzeni.
2. Oddzielić analizę położenia `z_t`, przesunięcia `z_t-z0`, przyrostu `v_t=z_{t+1}-z_t` i przyspieszenia `v_t-v_{t-1}`. PCA pozycji może dobrze opisywać różnice obrazów i słabo zachować małe ruchy. Porównać PCA stanów z bazą przyrostów; podać zachowaną energię i błąd rekonstrukcji przyrostów, nie tylko wariancję pozycji.
3. Równolegle analizować rowspace fc2 (maksymalnie 192 współrzędne), jego kernel i pełne 768D. To stałe projekcje tego checkpointa. Wpływ na zadanie nadal weryfikować przez oryginalny downstream, a nie tylko błąd wektorów.
4. Dla każdej pary kolejnych kroków liczyć normę prędkości, cosinus kolejnych przyrostów, składową radialną ku/od z0, długość drogi, stosunek drogi do ruchu netto i liczbę istotnych zwrotów. Przy normach poniżej ustalonej tolerancji kierunek jest „nieokreślony”, nie losową jednostkową etykietą.
5. Badać wspólność również jako zgodność **reguły warunkowej**, a nie koniecznie równoległość surowych wektorów. Ta sama poprawka funkcjonalna, np. podniesienie właściwego logitu lub poszerzenie ramki, może wymagać innych kierunków zależnie od z0 i lokalnego Jacobianu głowy.

### 8.3. Rozdzielić zegar kroków od zegara zdarzeń

Przygotować dwa widoki bez mieszania ich znaczenia:

- **Zegar absolutny t:** pokazuje efekty schedule, np. skok lr 100→101. Porównywać osobno fit i min_norm oraz sukcesy/porazki. Nie wyrzucać porażek z opisu, lecz nie udawać, że mają fazę min_norm.
- **Zegar zdarzenia tau=t−first_success:** synchronizuje przełączenie celu w udanych przypadkach. To analiza warunkowa po wystąpieniu sukcesu, więc nie dowód zachowania całej populacji ani cecha dostępna przy inferencji z nowego obrazu.

Nie skalować arbitralnie czasu każdego obrazu tak, aby początek i after udawały tę samą prędkość. Jeśli używać phase-normalized time lub dopasowania trajektorii, zachować osobny raport w prawdziwych krokach. Analizować rozkłady i kwantyle per film, a nie tylko średnią pozycji wszystkich tokenów. Średnia chmury nie jest trajektorią każdej cząsteczki.

W min_norm naturalną prostą podłogą jest powrót ku z0. Aby deklarować nową wspólną dynamikę, operator powinien wyjaśniać coś ponad tę znaną konstrukcję straty i pamięci optymalizatora. Sensowne kontrole zachowują długość, fazę, lr i rolę tokenu; nie tasują dowolnie obrazów w fałszywy timeline.

### 8.4. DMD, DMDc, EDMD/Koopman: kiedy problem jest identyfikowalny

Budować jawne zbiory par `(i,j,t)→(i,j,t+1)`. Dla DMDc:

\[
Y=AX+BU+b\mathbf 1^T+E.
\]

Nie łączyć końca jednego tokenu z początkiem następnego. Wiersz/kolumna zależy od biblioteki, ale tożsamość pary musi pozostać jawna. Dwa segmenty długości 100 i 400 nie są próbkami jednakowego kroku. `after` o indywidualnym czasie nie jest kolejną wspólną klatką.

Rozdzielenie A i B wymaga dostatecznego rzędu i uwarunkowania macierzy `[X;U;1]` oraz danych obejmujących właściwe warunki. Gradient zależny od stanu/celu jest sterowaniem w pętli zamkniętej. Jeżeli w danym zakresie `u=Kx`, dane mogą identyfikować tylko `A+BK`; osobne A/B nie są wtedy jednoznaczne. Ridge wybiera rozwiązanie numeryczne, nie tworzy brakującej informacji przyczynowej. Trzeba raportować singular values, effective rank, korelacje stanu z wejściem i stabilność względem ridge/ranku.

Ponadto system zależy od fazy, czasu, GT i residual. Jeden autonomiczny operator z_t→z_t+1 może być niewystarczający. Sensowne opcje to model rozszerzonego stanu, operator warunkowany fazą/obrazem lub modele lokalne. Wariant „dodajmy Adam” musi uwzględniać nieliniowy preconditioner, czas, epsilon i clamp; sama PCA surowych m1/m2 nie jest odpowiednikiem dynamiki Adama.

Liniowe DMD jest modelem przejść w wybranych obserwablach. EDMD dopasowuje operator w bogatszym słowniku funkcji; skończony słownik nie gwarantuje domknięcia ani stabilnego rolloutu. „Koopman istnieje” nie oznacza, że da się go wiarygodnie zidentyfikować z 16 obrazów i bez tokenów celowych. Wspólny wzorzec musi powtarzać się na nowych filmach i lepiej przewidywać przyrosty/efekty niż proste kontrole.

Podstawa metod: Proctor, Brunton i Kutz opisują modelowanie z użyciem stanów i wymuszeń w [Dynamic Mode Decomposition with Control](https://arxiv.org/abs/1409.6358). Wnioski o rzędzie macierzy, feedbacku i brakujących zmiennych są tutaj zastosowaniem algebry identyfikacji do odczytanego kodu, a nie wynikiem tej pracy na SmallViT.

### 8.5. Co oferują sprawdzone biblioteki

Odczytano źródła pierwotne 9 października 2026; niczego nie instalowano:

| Narzędzie | Właściwe zastosowanie tutaj | Czego nie rozwiązuje automatycznie |
|---|---|---|
| deeptime DMD | Fit jawnych par opóźnionych stanów, kontrolowana redukcja i operator | Nie odtwarza brakujących kroków i nie wytwarza GT-free korektora |
| deeptime EDMD | Porównanie liniowych obserwabli z nieliniowym słownikiem, przy tym samym train/test | Bogaty słownik nie zastępuje warunkowania, liczby obrazów ani walidacji rollout |
| deeptime VAMP | Analiza skorelowanych w czasie obserwabli przez operator/covariance, także bez założenia równowagi; dostępne aktualizacje kowariancji partiami | Brak stacjonarnej dystrybucji nie blokuje samego operatora; zmiana prawa aktualizacji i nieobserwowany stan nadal wymagają świadomego modelu |
| PyDMD DMDc | Referencyjne dopasowanie dynamiki z kontrolą, znanym lub estymowanym B i wybranym rankiem | API `fit(X,I,B=None)` nie chroni przed sztucznymi przejściami po sklejeniu trajektorii i nie usuwa leakage wejść |

Źródła: [deeptime DMD](https://deeptime-ml.github.io/latest/notebooks/dmd.html), [deeptime EDMD](https://deeptime-ml.github.io/latest/notebooks/edmd.html), [deeptime VAMP API, w tym partial_fit](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.VAMP.html), [PyDMD DMDc API](https://pydmd.github.io/PyDMD/dmdc.html). Opis pamięci i skalowania Adama: Kingma i Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).

Warto użyć biblioteki jako niezależnej implementacji kontrolnej po naprawie kontraktu danych i skalowania. Sam import deeptime/PyDMD nie czyni wcześniejszego k-means temporalnym i nie gwarantuje przewagi nad NumPy.

### 8.6. Podłogi prostsze niż Koopman i zasady oceny

Porównać najpierw: persistence; stały przyrost fit tylko na train; fazowy model kurczenia `z_next−z0=a_phase*(z−z0)`; model przyrostu zależny od z0/pozycji/kontekstu; oraz analityczny replay Adam **wyłącznie jako kontrolę teacherowej osi czasu**. Model przyrostu powinien mieć domyślną tożsamość: `z_next=z+f(...)`, aby nie musiał od nowa odtwarzać dużych stałych różnic między obrazami.

Następnie DMD z ridge dobieranym w skalowanych cechach, DMDc z odpowiednio skalowanymi blokami i jawnym czasem/fazą, a dopiero później EDMD lub model nieliniowy. Dobór ranku/skali/kary tylko na walidacji train. Nie dopasowywać od razu gęstego operatora całej chmury 460 800×460 800: to 212 miliardów parametrów i daleko za mało niezależnych warunków. Model współdzielony per token z kontekstem lub operator niskiego rzędu jest inną, testowalną hipotezą.

Ocena: RMSE przyrostów i pełnych stanów, cosinusy tylko przy niezaniedbywalnym ruchu, rowspace/nullspace, rollout 1/5/20 kroków, zgodność przełączeń, błędy osobno w fit/min_norm i dla celu/pozostałych. Długi rollout ze znanymi przyszłymi wejściami oznaczyć jako teacher-forced, a nie samodzielny predyktor. Split po całych filmach; nie losować poszczególnych kroków ani tokenów tego samego obrazu między train/test. Wnioskowanie o niepewności po filmach, nie po 64 000 silnie zależnych parach.

## 9. Jak przejść od wzorca do użytecznej korekty bez GT

Wzorzec opisowy nie jest jeszcze regułą decyzyjną. Należy osobno sprawdzić **co warto zrobić** i **czy da się wybrać to działanie ze stanu naturalnego**.

Korektor może mieć formę:

\[
\Delta_{i,j}=a_\theta(z^0_{i,j},c_i,p_j)\,
\frac{d_\theta(z^0_{i,j},c_i,p_j)}{\|d_\theta(z^0_{i,j},c_i,p_j)\|+\epsilon},
\]

albo miękkiej mieszanki kierunków/reguł z bramką `p(r|Z0,context)`. Norma i kierunek są osobnymi zmiennymi; nie uśredniać wektorów o skrajnie różnych skalach bez kontroli. `c_i` w deploy to dostępny kontekst naturalny: lokalne sąsiedztwo, pooled cechy, naturalne logity/ramki, ewentualnie `q0` i niepewność. Nie GT, nie first_success, nie gradient oracle ani z_after. Patch pozostaje indeksem 20×30, ale wspólna funkcja może reagować odmiennie na ten sam patch w różnych obrazach.

Proponowana droga:

1. Z nauczyciela wyodrębnić **udane** zachowania i osobno porażki. Zachować porażki do diagnozy osiągalności i ryzyka, nie traktować ich automatycznie jako dobrych etykiet ruchu. W kontrolach sprawdzić wagowanie po obrazie, ograniczenie norm i oddzielną wagę roli celu/pozostałych.
2. Wzorzec może dotyczyć zmiany odpowiedzi głowy, np. poprawy ramki i rankingu, a nie jednego dokładnego endpointu 768D. Rozwiązania równoważne downstream nie muszą być zmuszane do zgodności we wszystkich null-directions.
3. Uczyć regułę wyboru ruchu ze z0+kontekstu, z pełnym podziałem filmów. Dla grup/reżimów raportować: liczność, rozrzut norm, średnią jednostkową i jej długość, parowe kąty, zadaniowy efekt i możliwość przewidzenia grupy z cech naturalnych. Kierunek grupy o małej zgodności oznaczyć jako nierozstrzygnięty, nie wymuszać średniej.
4. Porównać statyczną korektę z krótką iteracyjną regułą. Wariant iteracyjny musi sam generować potrzebny sygnał bez GT; nie podawać mu w teście zapisanych g_t i faz teacherowych. Dla wielu kroków oceniać stabilność poza stanami odwiedzonymi przez teachera.
5. Dopuścić ruch zerowy i kalibrować skalę na walidacji. Zabezpieczenie od niepewnych korekt nie jest gwarancją poprawy każdego obrazu, lecz porównywalnym wariantem reguły.

Średnia końcowych optimów nie jest optimum wspólne. Formalnie problem A to minimalizacja błędu odtwarzania delt, podczas gdy Hubble bada bezpośrednią optymalizację wspólnej delty przez stratę detekcji:

\[
\frac1N\sum_i \arg\min_\Delta L_i(\Delta)
\ne \arg\min_\Delta\frac1N\sum_i L_i(\Delta)
\]

w ogólności, zwłaszcza dla różnych aktywnych ograniczeń i nieliniowego ogona. Niepowodzenie A nie rozstrzyga wyniku Hubble'a. W tym audycie nie czytano ani nie zmieniano kodu/stanu jego bieżącej rodziny i nie raportuję jej nieznanych wyników.

### Test rozstrzygający — propozycja, niewykonana

Zamrozić K1000, dokładny hook, preprocessing, decoder, manifest i podział **train/validation/test po filmach**. Zewnętrzne filmy, jeśli to możliwe, powinny być również poza danymi użytymi do wytrenowania K1000. Ten ostatni warunek trzeba sprawdzić z manifestami, nie zgadywać z nazwy holdout.

| Ramię | Pytanie rozstrzygane | Informacja w inferencji |
|---|---|---|
| Naturalny K1000 | Punkt odniesienia | Obraz |
| Oryginalne A/B | Reprodukcja punktu wyjścia | z0, pozycja, kontekst zgodnie z wariantem |
| A success-only, skala dobrana na validation | Czy dominacja porażek była główną przeszkodą? | Stała delta per patch |
| A/B z kontrolą norm i progiem „brak ruchu” | Czy zawodzi magnituda i kierunki małych reszt? | Cechy naturalne |
| Prosty ridge/mały adapter warunkowy endpointu | Czy warunkowanie wystarcza bez modelowania czasu? | z0+kontekst, ewentualnie q0/naturalne outputy |
| Korektor z wzorców temporalnych | Czy czas wnosi coś ponad endpointy przy równym dostępie do danych? | Ten sam zestaw cech naturalnych, bez teacherowego g/fazy |
| Bezpośrednia wspólna delta Hubble'a | Czy źle wyznaczono stałą deltę, czy potrzebne jest warunkowanie? | Wynik oddzielnego biegu Hubble'a, nie zmieniany tutaj |
| Oracle z GT | Jak duży jest dostępny odstęp jakości na tej samej próbce? | GT; wyłącznie oddzielna górna referencja |

Dodatkowa diagnostyka B: prawdziwa etykieta discovery na testowych deltach vs przypisanie ze z0. Jeśli nawet oracle-label B nie pomaga, problem leży także w tablicach/średnich; jeśli pomaga, a normalne B nie, główną barierą jest reguła przypisania. Wyniku z oracle-label nie wolno przedstawiać jako deployable ani dobierać nim końcowych hiperparametrów testu.

Raportować mAP50:95/AP50/AP75, confidence i IoU top-1, odsetek obu90, recall/FP i pogorszenia per obraz/film. Wszystkie zmiany względem tego samego naturalnego baseline; bootstrap grupowany po filmie i przedziały ufności. Sprawdzić też kalibrację score, aby wzrost confidence nie zastąpił poprawnej lokalizacji. Zachować koszt i budżet modeli; większy temporalny model nie powinien dostawać dodatkowego GT w teście.

Interpretacja możliwych wyników:

- Success-only pomaga: wskazuje na zanieczyszczenie teacherowych targetów, nie automatycznie na wartość Koopmana.
- Bezpośrednia wspólna delta pomaga, A nie: problemem było wyznaczenie delty, nie koniecznie sama stałość.
- Prosty warunkowy korektor pomaga, temporalny nie lepiej: przydatne jest warunkowanie, brak dowodu dodatkowej wartości czasu.
- Temporalny pomaga ponad prosty korektor na nowych filmach: dowód dodatkowej użyteczności wyuczonych regularności, z zakresem określonym splitami i budżetem.
- Żaden transfer nie pomaga, oracle pomaga: luka teacher→student pozostaje. To nie dowód niemożliwości wszelkiej wspólnej korekty.

## 10. Co trzeba dodatkowo zapisać i ile to kosztuje

Pełny zapis stanów float32 to dokładnie:

\[
1500\cdot501\cdot600\cdot768\cdot4
=1\,385\,164\,800\,000\ \text{bajtów}
\]

czyli **1,385165 TB dziesiętnie**, około **1,25980 TiB**, bez gradientu, pamięci, metadanych i narzutów plików. Jeden pełny wspólny stan to 2,7648 GB. Kompresja nie ma gwarantowanego współczynnika i nie może być założeniem preflightu.

| Wariant zapisu | Surowy rozmiar | Utrata/ograniczenie |
|---|---:|---|
| Pełne z, float32 | 1,385 TB | Zachowuje każdy wektor; nie wystarcza samo do pełnego odtworzenia celu/pamięci |
| Pełne z, float16 | 692,582 GB | Kwantyzacja niszczy małe przyrosty; z[t+1]−z[t] może zniknąć |
| PCA32 każdego tokenu/kroku, float32 | 57,7152 GB | Traci 736 kierunków i kąty niewidoczne w projekcji |
| PCA64 każdego tokenu/kroku, float32 | 115,4304 GB | Więcej kierunków, nadal brak pełnej geometrii |
| 16 obrazów, wszystkie 600 tokenów, 768D FP32 | 14,7751 GB | Pełna współdynamika w małej próbie; nie cała populacja |
| 64 obrazy, wszystkie tokeny, 768D FP32 | 59,1004 GB | Lepsza próba, nadal ograniczona liczba filmów i przypadków |
| Dotychczasowe 128 trajektorii z w FP32 | 197,001 MB | Brak targetów i zdecydowanej większości tokenów |

Pełne g, m1 i m2 dla wszystkich tokenów przez 500 przejść dodałyby **4,1472 TB**, razem z z około **5,532365 TB**. Nie trzeba zapisywać tego naiwnie, jeśli pytanie da się odpowiedzieć inaczej.

Minimalny poprawny kontrakt dodatkowej rejestracji:

- Własny manifest i hashe checkpointa/kodu; image_id, film_id, token_id, t, dtype, wspólna baza/skala i wersja preprocessing.
- Wszystkie 600 tokenów co krok przynajmniej dla warstwowanej próby obrazów. W panelu zawsze target, naturalny top-1, konkurenci, patche obiektu i tła oraz kontrolne pozycje. Role wyznaczone GT zapisać jako etykiety **offline**, nie cechy inference.
- `z_t`, `z0`, `q0`/residual potrzebny do dokładnego downstream. Dla pełnego replay: g przed krokiem, m/v przed krokiem lub równoważny checkpoint pamięci, licznik Adama, betas, epsilon, lr, clamp. Precyzyjnie rozróżnić indeks stanu i indeks przejścia.
- Faktycznie aktywna strata i jej składniki, a nie wyłącznie fit_objective po przejściu do min_norm; first_success, aktualny success, target/top token, mapy confidence i ramki, normy gradientów składników w małej próbie.
- Końcowe before/after/delta, **first-success state** i stan przełączenia. Sam after nie zastępuje tych stanów. Wersja sukcesu powinna zachować wynik przed i po ewentualnej minimalizacji.

Oszczędne alternatywy:

1. **Streaming fit operatora/kowariancji:** mając bieżące i poprzednie stany, aktualizować statystyki par, osobno dla train, faz i ról. Dla liniowej regresji można zachować gram/cross-gram zamiast wszystkich trajektorii. Traci się możliwość arbitralnego ponownego klastrowania i oglądania pojedynczych zwrotów; PCA/cechy trzeba ustalić wcześniej albo przewidzieć drugi przebieg. Nie wolno fitować wspólnej bazy na testowych filmach.
2. **Stałe PCA + projekcje przyrostów + normy/iloczyny w 768D:** oprócz współrzędnych zapisywać normę v_t, normę netto, iloczyn v_t·v_t−1, energię row/null i residuum rekonstrukcji. Normy dwóch wektorów same nie odtwarzają kąta; iloczyn lub cosinus trzeba zapisać osobno. Tracimy pełny kierunek w odrzuconej podprzestrzeni i możliwość jego późniejszego odzyskania.
3. **Współrzędne rowspace fc2 + mała próba pełnych 768D:** przy niezmiennym W2 i residual jest to dokładny opis wpływu liniowej projekcji na downstream, ale nie całego procesu Adam/clamp ani geometrii naturalnego latentu. 192D nadal kosztuje około 346,291 GB dla wszystkich kroków.
4. **Próbka pełna 768D + szeroki panel skompresowany:** najbardziej praktyczny kompromis do pierwszego badania. Próbkowanie po filmach, trudnościach, fazach i rolach; sama równomierna siatka bez celu powtórzyłaby wadę obecnego panelu.
5. **Deltowanie/kompresja bezstratna:** może zmniejszyć rozmiar, lecz trzeba zmierzyć współczynnik na reprezentatywnych strumieniach. Zapis samego float32 przyrostu nie gwarantuje bitowego odtworzenia z przy kumulacji zaokrągleń; potrzebne okresowe pełne stany i pomiar dryfu lub kodowanie bezstratne bitów.

Nie uruchomiono ponowienia. Budżet czasu zapisu i wolne miejsce wymagają oddzielnego zlecenia i aktualnego pomiaru, nie historycznej informacji o dysku.

## 11. Minimalny kolejny krok i kryterium powodzenia

Najmniejszy sensowny krok **bez ponawiania oracle** to kontrola już istniejących endpointów: success-only i kontrola norm, z tymi samymi foldami filmów, oraz prosty korektor zależny od z0+kontekstu. To odróżni problem teacherowych etykiet od rzeczywistej potrzeby modelu czasu. Można to zestawić z oddzielnym wynikiem Hubble'a, gdy będzie dostępny, bez ingerencji w jego pracę.

Równolegle dla pytania temporalnego: na obecnych 128 trajektoriach porównać persistence, fazowy powrót do z0, poprawnie skalowany model przyrostów i operator rozszerzonego stanu. Jest to nadal test **tokenów niecelowych**. Pełna odpowiedź o wspólnym ruchu naprawiającym narzędzie wymaga następnie autoryzowanego zapisu pełnych 600 tokenów co krok w ograniczonej, warstwowanej próbie obrazów, obejmującej target i konkurentów. Nie ma potrzeby zaczynać od 1,385 TB.

Kryterium powodzenia nie brzmi „znaleźliśmy klastry” ani „R²≈1”. Brzmi: reguła dostępna bez GT w inferencji poprawia detekcję na odłożonych filmach ponad naturalny K1000 i porównywalny prosty korektor, z kontrolą pogorszeń i niepewności. Wynik 90 procent na każdym obrazie nie jest obiecany ani poparty obecnymi danymi.

## 12. Weryfikacja i komplet artefaktów

Wykonane: odczyt instrukcji i stanu Git, retrospekcja raportów, porównanie kodu/inventory/Eden, odczyt zakończonych jobów, opisowa diagnostyka lokalnych norm i telemetrii, diagnostyka CPU 16 istniejących NPZ, replay Adama, ponowne zastosowanie **istniejącego** DMD/DMDc oraz przegląd primary dokumentacji. Niewykonane: nowy trening, nowe joby, fit nowego DMD, ablacje detekcji, generowanie obrazów, zmiany wizualizacji i modeli.

Kontrola końcowa: wszystkie 44 lokalne odnośniki raportu wskazują istniejące pliki, numery linii mieszczą się w plikach; kopie źródeł mają zgodne SHA256. Wszystkie 16 hashy odczytanych NPZ panelu zgadza się z metadanymi wcześniej zweryfikowanej projekcji. Składnia czterech własnych skryptów jest poprawna. Końcowy status Git pokazuje wyłącznie nowe katalogi tej rodziny w objętym kontrolą zakresie; `git diff --check` w tym zakresie nie zgłosił błędów (pliki pozostają nieśledzone, niczego nie stage'owano ani nie commitowano).

Odczytowy lint rejestru Grant wykazał cztery zastane błędy (brak legacy source i dwa odwołania do niego, nieobsługiwany stan AU31). Nie dotyczą tej rodziny i nie zostały naprawione poza ownership. Nie ma nowych AU ani zmian ich statusu. Retrospekcja niniejszego raportu rozlicza twierdzenia, nie wdrożone naprawy pipeline; **NAPRAWA_POTWIERDZONA=0, WDROŻONE_NIEZWERYFIKOWANE=0** w zakresie tego audytu.

Raport zawiera siedem lokalnych ustaleń P1 (A-01…A-07), tabelę dziewięciu rozliczonych interpretacji i żadnego nowego globalnego AU. Trzy z tych interpretacji oznaczono jako obalone w podanym zakresie: całkowity brak DMD, pełny timeline wszystkich cząsteczek oraz przeniesienie liczby J128 na oracle1500. Nie są to trzy naprawy kodu. Żaden proponowany nowy korektor lub test generalizacji nie został wykonany.

Źródła własnych obliczeń:

- [DIAGNOSTYKA_LOKALNA.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/DIAGNOSTYKA_LOKALNA.json): cała populacja poprzez istniejące normy, role, fazy, rozkład patchy, skala A i dane B.
- [DIAGNOSTYKA_EDEN.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/DIAGNOSTYKA_EDEN.json): pełne 128 trajektorii, replay, fazowy wynik istniejącego DMDc, nullspace endpointów 16 obrazów.
- [PROVENANCE_ODCZYTU.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/PROVENANCE_ODCZYTU.json): SHA256 kodu, zgodność z historycznym inventory i lokalnym worktree; dokładne kopie kodu/logów pod `snapshot_eden/`.
- [README diagnostyk](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_k1000_analiza_rezimow_astra_20261009/README.md): odtwarzanie i ograniczenia małych skryptów.
- [PODSUMOWANIE_DIAGNOSTYK.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/PODSUMOWANIE_DIAGNOSTYK.json): agregaty, hash wejść, sprawdzenie snapshotów i lokalnych odnośników raportu.

Nie zmieniono żadnego historycznego wyniku. Raport koryguje jego interpretację i wskazuje testy, które dopiero mogą rozstrzygnąć transfer wspólnej dynamiki.
