# Nowość i ryzyko wyprzedzenia — wyniki „mikroskopu latentnego” (mały ViT, końcówka narzędzia, katarakta)

Data: 2026-10-21. Model: d = 192, 600 tokenów łat, 6 bloków, trening od zera, 806 checkpointów do kroku 65640 (≈ 60 epok).
Źródła: Undermind (workspace „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”), WebSearch/WebFetch. Scite nieużywany (limit do 13.11.2026).

Zasady: każdą kluczową pracę sprawdzono co najmniej na poziomie streszczenia (Undermind `get_paper_info` lub WebFetch). Gdzie streszczenia nie dało się pobrać, zaznaczam to wprost. **[preprint]** = tylko arXiv/ResearchGate/brak recenzji wg metadanych. DOI podaję wyłącznie te zwrócone przez Undermind albo wydawcę; brak = „nie znaleziono” (podaję wtedy link arXiv).

Nowe deep searche z tej sesji (Undermind):
- [Ewolucja tokenów łat obiekt vs tło w czasie treningu](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Training%20time%20evolution%20of%20object%20versus%20background%20patch%20token%20representations%20in%20vision%20transformers%20across%20checkpoints)
- [DiProPerm / permutacje z refitem osi dla reprezentacji sieci](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Permutation%20tests%20with%20refitted%20projection%20directions%20DiProPerm%20applied%20to%20neural%20network%20representations)
- [Iluzje rzutów i przekrojów](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Interpretability%20illusions%20from%20low%20dimensional%20projections%20and%20slices%20of%20weights%20activations%20and%20vector%20fields)
- [Tokeny jako cząstki w czasie treningu, MLP jako siła](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Token%20representations%20as%20interacting%20particles%20evolving%20over%20training%20time%20with%20MLP%20neurons%20as%20forces)
- [Interpretowalność modeli narzędzi chirurgicznych w MICCAI/IJCARS/MedIA](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Interpretability%20and%20internal%20representation%20analysis%20of%20surgical%20instrument%20deep%20learning%20models%20in%20medical%20imaging%20venues)

Wykorzystane wcześniejsze searche (ten sam workspace): „Tokens and representations as interacting particles…”, „Prior art for tracking token representations across training as particle trajectories…”, „Prior art check for specific measured findings on opposing MLP neuron pairs…”, „Antipodal feature pairs…”, „Hidden neurons merging into near duplicate copies…”, „Dynamics-respecting dimensionality reduction…”, „Projecting high-dimensional velocity vectors…”, „Rotational curl components…”, „Validating class separation in supervised projections…”, „Training time MLP attention readout mechanisms… novelty audit”.

---

## Wynik 1. Stany tokenów narzędzia vs tła w strumieniu rezydualnym przez cały trening

Teza: szybka relaksacja (~30 epok), potem segregacja narzędzia od tła i spadek PR chmury tokenów 10,6 → 3,2 długo po plateau IoU; oś czasu = trening, nie głębokość.

**Werdykt: CZĘŚCIOWO OPISANE.** Ogólny wzorzec („szybkie dopasowanie, potem wolna kompresja/kolaps reprezentacji po nasyceniu metryki”) jest dobrze znany na poziomie klas i obrazów. Nie znalazłem pracy, która śledzi chmurę **tokenów przestrzennych podzieloną na obiekt/tło w jednym obrazie** przez setki checkpointów treningu od zera, ani pracy z tego typu pomiarem w detekcji czy segmentacji.

Najbliższe prace:
1. **Shwartz-Ziv & Tishby 2017**, *Opening the Black Box of DNNs via Information* — arXiv:1703.00810, DOI: nie znaleziono **[preprint]**. Opisują fazę „szybkiego dryfu”, po której przychodzi „stochastyczna relaksacja” i kompresja reprezentacji, gdy błąd treningowy jest już mały. To wprost poprzednik naszej narracji o dwóch fazach. **Uwaga:** krytyka **Saxe i in. 2018/2019**, DOI 10.1088/1742-5468/ab3985, pokazuje, że faza kompresji zależy od nieliniowości (nasycające tanh tak, ReLU nie) i nie jest przyczynowo związana z generalizacją. Trzeba to zacytować i nie twierdzić, że spadek PR „tłumaczy” generalizację.
2. **Varam i in. 2025**, *Class Separation Dynamics in Vision Transformers: An Empirical Study*, IEEE Access, DOI 10.1109/ACCESS.2025.3602733. W ViT „fuzziness” separacji klas maleje wykładniczo z głębokością, a w ostatniej warstwie także z epokami. Dotyczy klasyfikacji całych obrazów (miara He & Su 2022, PNAS, DOI 10.1073/pnas.2221704120), więc nie ma rozdziału obiekt/tło wewnątrz obrazu (sprawdzono README repozytorium, kod nieprzeczytany).
3. **Kapil & Upla 2026**, seria *Transformer Geometry Observatory* TGO-I/II/III — TGO-I DOI 10.48550/arXiv.2606.19249 **[preprint]**, TGO-II 10.48550/arXiv.2607.02386 **[preprint]**, TGO-III 10.13140/RG.2.2.26440.07689 **[preprint, ResearchGate]**. ViT-S/16 na ImageNet-100 przez cały trening: PR, effective rank i ID **rosną** (TGO-I), a CKA/SVCCA i kowariancję tokenów śledzą w czasie (TGO-II). Kierunek jest przeciwny do naszego (u nas PR spada), co daje argument za specyfiką zadania „mały obiekt + tło”. To aktywna seria, więc **ryzyko wyprzedzenia przez kolejne części (TGO-IV…)**.

Uzupełniające (cytować w tle):
- Papyan, Han, Donoho 2020, PNAS, DOI 10.1073/pnas.2015509117 (neural collapse w terminal phase, po zerowym błędzie).
- Zhong i in. 2023, CVPR, DOI 10.1109/CVPR52729.2023.01873: neural collapse w segmentacji na poziomie pikseli. Kontekst i nierównowaga klas łamią ETF. Analiza stanu końcowego, bez trajektorii przez checkpointy.
- Tan i in. 2024, IJMLC, DOI 10.1007/s13042-024-02244-x: przejście fazowe ekspansja → kompresja ID reprezentacji w czasie treningu.
- Patel & Shwartz-Ziv 2024, DOI 10.48550/arXiv.2410.07687 **[preprint]**: lokalny rząd maleje w końcowej fazie treningu (MLP).
- Sharon & Dar 2024, arXiv:2405.17377, DOI: nie znaleziono **[preprint]**: CKA i sondy liniowe przez cały trening, także długo po dopasowaniu. ViT ma zsynchronizowaną dynamikę warstw.
- Raghu i in. 2017 (SVCCA, NeurIPS), DOI: nie znaleziono, link: https://www.semanticscholar.org/paper/a3f7a30abe44424e5ef8348a02cc103237ac5210 (zbieżność „bottom-up” w czasie treningu).
- Alain & Bengio 2016, arXiv:1610.01644, DOI: nie znaleziono **[preprint]**: sondy liniowe.
- Järve i in. 2026, DOI 10.1145/3799682.3841045: ewolucja i migracja cech SAE w ViT po osi warstwa × epoka, tylko token CLS.
- Su 2026, DOI 10.48550/arXiv.2605.23868 **[preprint]**: degradacja reprezentacji gęstych łat przy przedłużonym treningu DINO. To kontrast: tam łaty tracą lokalność.
- Shin i in. 2025, *Self-Guided MAE*, arXiv:2507.19773 **[preprint]**: klasteryzacja na poziomie łat pojawia się wcześnie w pretreningu MAE. Według streszczenia, sposób śledzenia przez checkpointy nie jest opisany.
- Recanatesi i in. 2019, arXiv:1906.00443 **[preprint]**: ekspansja/kompresja wymiarowości (po głębokości).
- Chou i in. 2025, DOI 10.48550/arXiv.2503.18114 **[preprint]**: geometria rozmaitości w czasie uczenia, „beyond lazy-rich”.

Co jest nowe (do sformułowania wąsko): (a) poziom tokenów przestrzennych z podziałem obiekt/tło w detekcji małego obiektu, (b) gęste próbkowanie 806 checkpointów, (c) ilościowe rozdzielenie „plateau IoU” i „dalszej segregacji/spadku PR”, (d) kierunek PR przeciwny do TGO-I.

---

## Wynik 2. Rozdział narzędzia od tła już w top-3 PCA; LDA ze skurczem dokłada margines; DiProPerm + walidacja blokowa po checkpointach

**Werdykt: CZĘŚCIOWO OPISANE (metodycznie), w zastosowaniu do reprezentacji sieci w czasie treningu — NOWE.** Nie znalazłem żadnego użycia DiProPerm na reprezentacjach sieci neuronowych (deep search Undermind i WebSearch). Nie znalazłem też połączenia „permutacja z refitem osi + CV blokowa po checkpointach”. Same składniki są standardem w statystyce, neuroobrazowaniu i chemometrii.

Najbliższe prace:
1. **Wei, Lee, Wichers, Li, Marron 2016** (DiProPerm), JCGS 25(2), DOI 10.1080/10618600.2015.1027773. Test dla HDLSS: projekcja na kierunek klasyfikatora liniowego, kierunek dopasowywany od nowa przy każdej permutacji. Zastosowania: mikromacierze. Pakiet R: Allmon, Marron, Hudgens, R Journal, DOI 10.32614/rj-2021-072.
2. **Valente i in. 2021**, NeuroImage, DOI 10.1016/j.neuroimage.2021.118145. Jedyna poprawna strategia permutacyjna to przeliczenie całej CV na każdym spermutowanym zbiorze. To bezpośrednie uzasadnienie dla refitu.
3. **Worley & Powers 2016**, Current Metabolomics, DOI 10.2174/2213235X04666160613122429. PCA jako wskaźnik wiarygodności OPLS-DA: separacja widoczna bez etykiet uwiarygadnia projekcję nadzorowaną. To dokładnie logika „top-3 PCA, sylwetka 0,85 bez etykiet → LDA dodaje tylko margines”. Podobnie Bevilacqua & Bro 2020, *Can We Trust Score Plots?*, Metabolites, DOI 10.3390/metabo10070278 (wykresy score’ów z CV zamiast z kalibracji).

Uzupełniające:
- Hewitt & Liang 2019, EMNLP, DOI 10.18653/v1/D19-1275: control tasks i selektywność sond. To odpowiednik permutacji etykiet dla sond w NLP.
- Kriegeskorte i in. 2009, Nat. Neurosci., DOI 10.1038/nn.2303 (double dipping); Maindonald & Burden 2005, ANZIAM J., DOI 10.21914/ANZIAMJ.V46I0.947 (selection bias na wykresach dyskryminacyjnych).
- Ahn & Marron 2010, Biometrika, DOI 10.1093/BIOMET/ASP084 (data piling: LDA w HDLSS daje pozorną idealną separację); Bickel, Kur, Nadler 2018, PNAS, DOI 10.1073/pnas.1801177115.
- Roberts i in. 2017, Ecography, DOI 10.1111/ECOG.02881 (CV blokowa dla danych zależnych); Winkler i in. 2015 (multi-level block permutation, NeuroImage, DOI nie pobrano: nie znaleziono w tej sesji).
- Waagen i in. 2019, arXiv:1907.04223 **[preprint]**: statystyka Henze–Penrose i permutacje Fishera do separacji klas między warstwami, także w czasie treningu. To najbliższy przykład permutacji dla separacji w reprezentacjach sieci, ale bez refitu kierunku i bez bloków.
- Chaibub Neto 2018, arXiv:1802.07714 **[preprint]**: płytkie klasyfikatory na cechach DNN kontra te same klasyfikatory na przetasowanych etykietach, przez epoki.

Wniosek: w artykule przedstawić jako **przeniesienie znanej procedury** (DiProPerm + CV blokowa) do analizy checkpointów, a nie jako nową metodę. Nowością jest wynik (rozdział jest nienadzorowany, LDA = margines), a nie test.

---

## Wynik 3. Wynik negatywny: PCA kolumn fc2 i przekroje 3D pola sił MLP fabrykują struktury

**Werdykt: CZĘŚCIOWO OPISANE.** Zasada, że rzut lub przekrój tworzy pozorną strukturę, jest dobrze ugruntowana (trajektorie treningu, neuronauka, single-cell). Nie znalazłem pracy, która pokazuje to konkretnie dla (a) PCA kolumn wag wyjściowych MLP jako „widoku 3D” i (b) wirów w przekrojach pola aktualizacji MLP z zerem Ginibre’a. Ilościowe zera (1−cos30° dla fałszywych par, odsetek zespolonych wartości własnych dla losowych macierzy rzeczywistych) to wkład własny, ale prosty. Ta część nadaje się raczej na sekcję metodyczną lub ostrzeżenie niż na główną tezę.

Najbliższe prace:
1. **Antognini & Sohl-Dickstein 2018**, NeurIPS, arXiv:1806.08805, DOI: nie znaleziono. PCA losowego błądzenia daje krzywe Lissajous. Rzuty trajektorii treningu ResNet/MLP przypominają błądzenie z dryfem.
2. **Shinn 2023**, *Phantom oscillations in PCA*, PNAS, DOI 10.1073/pnas.2311420120. Gładkość i przesunięcia dają fałszywe oscylacje w PC, a zwykła CV ich nie wykrywa.
3. **Edelman, Kostlan, Shub 1994**, JAMS, DOI 10.1090/S0894-0347-1994-1231689-0. Oczekiwana liczba rzeczywistych wartości własnych losowej macierzy gaussowskiej n×n wynosi ~√(2n/π), więc większość widma jest zespolona. To formalne źródło zera dla „wirów”. Uzupełnia je Edelman 1997, JMVA (DOI w tej sesji nie pobrano).

Uzupełniające:
- Elsayed & Cunningham 2017, Nat. Neurosci., DOI 10.1038/nn.4617 (struktura populacyjna jako produkt uboczny prostszych cech; kontrole TME). Lebedev i in. 2019, Sci. Rep. (krytyka rotacji jPCA; DOI nie pobrano).
- Bolukbasi i in. 2021, *An Interpretability Illusion for BERT*, arXiv:2104.07143 **[preprint]**; Makelov i in. 2023, DOI 10.48550/arXiv.2311.17030 **[preprint]** (iluzja patchowania podprzestrzeni).
- Bickel, Kur, Nadler 2018, PNAS, DOI 10.1073/pnas.1801177115 (projection pursuit znajduje strukturę w szumie).
- Li i in. 2018, *Visualizing the Loss Landscape*, arXiv:1712.09913 (filter normalization: wybór kierunków przekroju zmienia obraz); Böttcher & Wheeler 2022, JSTAT (kierunki hesjanowe; DOI nie pobrano).
- Gigante i in. 2019, M-PHATE, arXiv:1908.02831 **[preprint]**; Rauber i in. 2017, IEEE TVCG, DOI 10.1109/TVCG.2016.2598838 (projekcje aktywacji przez trening; streszczenie znane tylko pośrednio ze strony projektu autorów). To techniki, nie ostrzeżenia.
- Zheng i in. 2023, Genome Biology (*Pumping the brakes on RNA velocity*): strzałki rzutowane na embedding są artefaktem kernela i wygładzania. Bardzo dobra analogia dla „pola sił w 3D” (DOI nie pobrano w tej sesji).
- Cai, Fan, Jiang 2013, JMLR (rozkład kątów losowych wektorów w wysokim wymiarze, zero dla „przeciwnych par”).

---

## Wynik 4. Interpretacja „tokeny = cząstki, neurony MLP = siły/lasery (wkład rzędu 1 jak ciśnienie promieniowania)”

**Werdykt: CZĘŚCIOWO OPISANE.** Obraz cząstek (tokeny jako cząstki) jest ugruntowany, ale **wyłącznie w czasie głębokości przy zamrożonych wagach**. W 2026 kilka preprintów włączyło FFN/MLP jako pole sterujące lub siłę, nadal w głębokości. **Nie znalazłem żadnej pracy, w której tokeny poruszają się w czasie treningu (kroki optymalizatora przez checkpointy) pod wpływem rozłożonych na neurony sił MLP, ani analogii laser/pułapka optyczna/ciśnienie promieniowania.** W klasycznym mean-field dla dwuwarstwowych sieci cząstkami są neurony, nie tokeny, a czas jest czasem treningu (Mei–Montanari–Nguyen 2018, PNAS, DOI 10.1073/pnas.1806579115; Chizat & Bach 2018).

Czy któraś praca dotyczy czasu treningu i MLP?
- **Isobe, Inoue, Imaizumi 2026**, *Training-Induced Escape from Token Clustering in a Mean-Field Formulation of Transformers*, DOI 10.48550/arXiv.2605.07772 **[preprint]**. Trenowany jest **tylko FFN** (liniowy w parametrach) i trening wypycha rozkład tokenów z klastrów w końcowych warstwach. To jedyna praca łącząca trening z FFN w obrazie cząstek, ale tokeny nadal ewoluują po głębokości. **Najbliższe ryzyko wyprzedzenia dla wyniku 4.**
- **Herty & Liu 2026**, arXiv:2608.25055, DOI: nie znaleziono **[preprint]**. Sprzęga miarę tokenów (głębokość) z miarą parametrów atencji (czas treningu, gradient flow w Wassersteinie). Tylko atencja.
- **Mudarisov, Burtsev, State 2026**, *Feed-Forward Steering in Transformer Residual Dynamics*, arXiv:2608.02071, DOI: nie znaleziono **[preprint]**. FFN jako lokalne „pole sterujące” na każdy token (składowa styczna kontra radialna) w LLM-ach. Głębokość, bez rozkładu na neurony.
- **Álvarez-López, Geshkovski, Ruiz-Balet 2026**, DOI 10.48550/arXiv.2601.21366 **[preprint]**: blok perceptronu w krajobrazie mean-field atencji. Maranzatto i in. 2026, arXiv:2609.36230 **[preprint]**: FFN jako sterowanie (kontrola) cząstek.

Klasyczne punkty odniesienia (koniecznie):
1. **Geshkovski, Letrouit, Polyanskiy, Rigollet 2023**, *The emergence of clusters in self-attention dynamics*, NeurIPS, DOI 10.52202/075280-2493, oraz *A mathematical perspective on Transformers*, DOI 10.48550/arXiv.2312.10794 **[preprint w chwili wpisu]**. Tylko atencja, wagi stałe, czas = głębokość.
2. **Lu i in. 2019**, *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View*, arXiv:1906.02762, DOI: nie znaleziono. FFN jako człon konwekcji w ODE po głębokości (Lie–Trotter → Macaron Net).
3. **Sander, Ablin, Blondel, Peyré 2022**, Sinkformers (AISTATS), arXiv:2110.11773, DOI: nie znaleziono. Atencja jako dyskretny gradient flow w Wassersteinie. Jedyna obserwacja „w czasie treningu” dotyczy macierzy atencji zbliżających się do podwójnie stochastycznych z epokami, nie trajektorii tokenów.
- Rigollet 2025, DOI 10.48550/arXiv.2512.01868 **[preprint]** (przegląd mean-field).
- Geva i in. 2022, DOI 10.48550/arXiv.2203.14680 (EMNLP): rozkład aktualizacji FFN na pod-aktualizacje pojedynczych wektorów wartości. To jest dokładnie „wkład rzędu 1 neuronu”, ale w przestrzeni słownika, bez interpretacji jako siły.
- Pashakhanloo & Koulakov 2023, ICML, DOI 10.48550/arXiv.2302.02563: dryf reprezentacji pod szumem SGD jako dyfuzja w czasie treningu (sieć dwuwarstwowa).

Wniosek: analogię laserową podawać jako **heurystykę opisową**, nie wynik. Nowość leży w zmianie osi czasu (trening) i w rozkładzie MLP na neurony, a nie w samym obrazie cząstek. Ten front rozwija się szybko (3–4 preprinty od maja do września 2026).

---

## Wynik 5. Przeciwne pary neuronów MLP per token z samego znaku GELU

Teza: kąt wkładów ≥150° powstaje tylko przez znak GELU przy prawie równoległych kolumnach W2 (żadna para kolumn nie jest antyrównoległa, maks. 142°). Pary to częściowe kopie z przesuniętymi progami i pojawiają się po głównym zysku IoU.

**Werdykt: CZĘŚCIOWO OPISANE (składniki), mechanizm i kolejność czasowa — NOWE.** Klasyczne „antypary” i „przeciwne osobliwości” to inny mechanizm: **przeciwne wagi wejściowe** przy nieparzystej aktywacji (±1 / tanh), co daje znoszenie wkładów. U nas wejścia są prawie identyczne, wyjścia równoległe, a przeciwieństwo bierze się z niemonotonicznego/ujemnego płata GELU. Duplikaty i kondensacja neuronów są znane. Kanały Martinellego dają parę z **przeciwnymi** wagami wyjściowymi, która w granicy staje się jednostką bramkowaną (GLU). To najbliższy funkcjonalny odpowiednik „częściowych kopii z przesuniętym progiem”.

Sprawdzone prace:
1. **Kwon & Kim 2005**, *Anti-pairing in learning of a neural network with redundant hidden units*, J. Phys. A 38(25):5627, DOI 10.1088/0305-4470/38/25/003 (streszczenie przez IOP). Committee machine z wagami ±1, metoda replik: nadmiarowe jednostki tworzą antypary (przeciwne wektory wag), których wkłady się znoszą. Przejście I rodzaju przy łamaniu symetrii permutacyjnej. **Inny mechanizm** (przeciwne wejścia, a nie znak aktywacji). Brak dynamiki gradientowej w głębokiej sieci.
2. **Guo i in. 2018**, *Stability analysis of opposite singularity in multilayer perceptrons*, Neurocomputing 282:192–201, DOI 10.1016/j.neucom.2017.12.018. **Streszczenia nie udało się pobrać** (ScienceDirect 403, brak w Undermind). Definicję potwierdzono przez wcześniejszą pracę tej grupy: Guo, Wei, Zhao, Zhang 2014, Control Theory & Technology 31(2):140–147 (DOI: nie znaleziono). Mowa tam o dwóch jednostkach ukrytych o prawie przeciwnych wagach, osobliwości wynikającej z symetrii permutacyjnej i wolnej dynamice/plateau. Inny mechanizm niż u nas (zakłada nieparzystą aktywację).
3. **Fukumizu & Amari 2000**, Neural Networks 13(3):317–327, DOI 10.1016/S0893-6080(00)00009-5 (streszczenie z preprintu autora). Punkt krytyczny sieci z H−1 jednostkami daje linie punktów krytycznych sieci z H jednostkami (duplikacja i podział wagi wyjściowej) oraz plateau. To teoretyczna podstawa „kopii”.
4. **Martinelli, van Meegen, Şimşek, Gerstner, Brea 2025**, *Flat Channels to Infinity in Neural Loss Landscapes*, DOI 10.48550/arXiv.2506.14951 **[preprint; wg późniejszych metadanych możliwie NeurIPS 2025 — nie zweryfikowano]**. Wagi wejściowe dwóch neuronów zbiegają się, a wyjściowe rozbiegają do ±∞, co daje GLU: σ(w·x) + (v·x)σ′(w·x). SGD/Adam trafiają tam często. **Najbliższa praca dla wyniku 5**, ale u nas kolumny W2 są równoległe, nie przeciwne.

Uzupełniające:
- Chen, Kunin, Yamamura, Ganguli 2023/2024, *Stochastic collapse*, JSTAT, DOI 10.1088/1742-5468/ad6429: SGD przyciąga do podsieci z neuronami znikającymi lub **redundantnymi (zduplikowanymi)**.
- Xu, Zhang, Zhou 2025, przegląd kondensacji, DOI 10.48550/arXiv.2504.09484 **[preprint]**: neurony kondensują w grupy o podobnych wyjściach, a liczba klastrów rośnie z treningiem.
- Gurnee i in. 2024, *Universal Neurons in GPT2*, TMLR, DOI 10.48550/arXiv.2401.12181: rodziny neuronów (m.in. predykcja/supresja), korelacje aktywacji między seedami. Nie opisuje par z równoległymi wyjściami i przeciwnym znakiem GELU.
- Kong i in. 2025, *Negative Pre-activations Differentiate Syntax*, DOI 10.48550/arXiv.2509.24198 **[preprint]**: ujemny region GELU jest aktywnie wykorzystywany, a jego rola rośnie w trakcie treningu. Wspiera tezę, że „znak GELU ma znaczenie”.
- Elhage i in. 2022 (*Toy Models of Superposition*), DOI 10.48550/arXiv.2209.10652 **[preprint]**: antypodalne pary cech. To pary cech, nie neuronów.
- Gerstner & Schütze 2025, DOI 10.48550/arXiv.2505.17936 **[preprint]** (neurony wzmacniające/osłabiające według cos(we, wy)).
- Casper i in. 2019 (*Frivolous Units*, AAAI); McGrath i in. 2023 (Hydra effect) — kompensacja i redundancja.

Wniosek: zacytować Kwon & Kim, Guo i in., Fukumizu & Amari jako „klasyczne przeciwne/zduplikowane jednostki” i **wyraźnie odróżnić mechanizm** (przeciwne wejścia przy aktywacji nieparzystej kontra równoległe wyjścia plus znak GELU). Martinelli 2025 jako najbliższy funkcjonalnie (para → jednostka bramkowana). To, że pary pojawiają się po głównym zysku IoU, nie ma precedensu w znalezionych pracach (jak w audycie z 2026-10-20).

---

## Dodatkowo: precedens w MICCAI / IPCAI / IJCARS / MIDL / MedIA (ostatnie ~3 lata)

Wniosek: **jest precedens dla interpretowalności modeli chirurgicznych, ale nie dla analizy dynamiki reprezentacji w czasie treningu.** Najbliższe:
1. **Kim, Kim, Kim 2025**, *SurgX: Neuron-Concept Association for Explainable Surgical Phase Recognition*, MICCAI 2025, LNCS 15969, s. 532–542. arXiv DOI 10.48550/arXiv.2507.15418 (DOI rozdziału Springer: nie znaleziono). Wiąże neurony z pojęciami chirurgicznymi (TeCNO, Causal ASFormer, Cholec80). **Najlepszy dowód, że analiza na poziomie neuronów przechodzi w MICCAI.**
2. **Holm, Ghazaei, Navab 2025/2026**, *ProtoFlow*, IJCARS, DOI 10.1007/s11548-026-03697-x. Interpretowalne prototypy dynamicznych grafów sceny (CAT-SG, **katarakta**).
3. **Shin i in. 2026**, *SurgCheck*, IJCARS, DOI 10.1007/s11548-026-03739-4. Diagnostyka, czy VLM faktycznie „patrzy” na obraz: praca typu „analiza zamiast nowej metody”.
4. **Konz i in. 2022**, *The Intrinsic Manifolds of Radiological Images and their Role in Deep Learning*, MICCAI 2022, DOI 10.1007/978-3-031-16452-1_65. Geometria/wymiar wewnętrzny jako główny wkład (radiologia, nie chirurgia; nieco ponad 3 lata). Kontynuacja: Konz & Mazurowski 2024, DOI 10.48550/arXiv.2408.08381 **[preprint]** (ID reprezentacji po warstwach, obrazy medyczne vs naturalne).

Inne: Cheng i in. *SurgXBench* (WACV 2026, DOI 10.1109/WACV61042.2026.00790); Li i in. 2026, J. Imaging, DOI 10.3390/jimaging12080374 (prototypy w segmentacji chirurgicznej). Brak prac o geometrii tokenów ViT dla narzędzi przez czas treningu. W IJCARS/MICCAI trzeba będzie mocno spiąć analizę z praktyką: okluzje, pewność detekcji, kryterium zatrzymania treningu, dobór danych.

---

## Podsumowanie ryzyka

| Wynik | Werdykt | Najbliższe (koniecznie cytować) | Ryzyko |
|---|---|---|---|
| 1. Dynamika tokenów narzędzie/tło w czasie treningu, spadek PR po plateau IoU | częściowo opisane | Shwartz-Ziv & Tishby 2017 (+ Saxe 2019), Varam 2025, Kapil & Upla TGO-I/II/III 2026 | średnie: aktywna seria TGO; kierunek PR odwrotny |
| 2. Separacja w top-3 PCA, LDA = margines, DiProPerm + CV blokowa | metodycznie znane, zastosowanie nowe | Wei i in. 2016 (DiProPerm), Valente 2021, Worley & Powers 2016 | niskie |
| 3. Fałszywe pary w PCA fc2, wiry = poziom Ginibre’a | częściowo opisane (zasada znana) | Antognini & Sohl-Dickstein 2018, Shinn 2023, Edelman i in. 1994 | niskie, ale mała siła nowości |
| 4. Tokeny = cząstki, neurony MLP = siły, czas treningu | częściowo opisane (głębokość); trening + MLP per neuron: nie znaleziono | Geshkovski i in. 2023, Lu i in. 2019, Isobe i in. 2026 | **najwyższe**: 3–4 preprinty 2026 dokładają FFN i trening |
| 5. Przeciwne pary przez znak GELU | składniki opisane, mechanizm nowy | Kwon & Kim 2005, Guo i in. 2018, Martinelli i in. 2025 (+ Fukumizu & Amari 2000, Chen 2023) | niskie–średnie |

Uwaga końcowa: werdykty opierają się na streszczeniach i metadanych, nie na pełnych tekstach. Przed submisją warto przeczytać pełne teksty: Varam 2025 (czy liczą coś na tokenach łat), TGO-II (kowariancja tokenów), Isobe 2026 i Martinelli 2025.
