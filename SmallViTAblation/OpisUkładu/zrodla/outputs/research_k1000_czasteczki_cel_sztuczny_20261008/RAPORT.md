# K1000: krytyczny przegląd cząsteczek, sztucznego celu i selekcji danych

Data: 2026-10-08. Status: zakończony research/spike, nie wykonany eksperyment produkcyjny.
Repozytorium: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation.
Zakres: frozen K1000, oracle 1500 obrazów, post-GELU ostatniego MLP, transfer korekty, SD, DINO/FAISS, wpływ paczek i dynamika AdamW.

## 1. Werdykt i najważniejsze rekomendacje

Koncepcja jest sensowna jako aparat diagnozy możliwości i ograniczeń głowy detektora oraz jako generator hipotez o korektach zależnych od kontekstu. Obecnie nie ma wystarczających podstaw, aby uznać sztuczny tor aktywacji za osiągalny tor uczenia albo podobieństwo DINO za miarę wartości danych. Nie rekomenduję uruchamiania całego kosztownego łańcucha przed przejściem poniższych bramek.

**Najsilniejszy nowy dowód:** historyczny sztuczny cel K1000 z eksperymentu J128 jest nieosiągalny przez zwykły forward tego zamrożonego modelu. Wynika to z analitycznych granic fc1 po LayerNorm, nie z niepowodzenia generatora. Naturalne górne granice po GELU wynoszą, zależnie od kanału, 7,0171–10,2368. Historyczne after osiąga 51,3532 i przekracza te granice w 15 661 z 460 800 współrzędnych. Dolna granica RMSE dowolnego naturalnego latentu do tego konkretnego celu wynosi co najmniej 2,50349. Przyszłe nowe cele pełne 768D wymagają własnego audytu; wynik nie obala ich z góry.

**Drugi dowód:** rzeczywista macierz fc2 ma kształt 192×768 i rangę numeryczną 192. Jej kernel ma wymiar 576. W historycznej korekcie J128 74,647 procent energii pełnej delty leży w tym kernelu. Ten składnik jest niewidoczny dla końcówki detektora przy zachowaniu wejścia residual. Może jednak dominować odległość do celu, klastrowanie i trudność inwersji.

**Trzeci problem strukturalny:** końcówka po interwencji jest tokenwise. Nie ma kolejnej uwagi mieszającej tokeny. Oracle sukces może polegać na bezpośrednim ustawieniu głowy dla właściwej komórki i tłumieniu konkurentów. To nie dowodzi odkrycia współdziałających semantycznych cząsteczek.

**Czwarty problem:** przy generowaniu nowego obrazu zmienia się także wejście gałęzi residual. Dopasowanie samego post-GELU z nie gwarantuje odtworzenia wyniku tail obliczonego z residual oryginalnego obrazu. SD musi być oceniany zwykłym pełnym forwardem, bez cache residual ze źródła.

Najpierw należy:

1. Uzupełnić cele o konieczne granice wynikające z norm2/fc1; mierzyć odległość od naturalnej rozmaitości i rozkład delty na rowspace/kernel fc2.
2. Traktować oracle jako zbiór dopuszczalnych zachowań i minimalnych korekt, nie jako jeden obowiązkowy wektor 600×768.
3. Porównać korektę średnią, korektor z kontekstem i prosty adapter uczony bezpośrednio stratą detekcji, na odłożonych filmach.
4. Zbudować krótką kontrolę rzeczywistego wpływu paczek ze zgodnego stanu wag, AdamW i BN. Ranking musi przewidywać użyteczność, nie tylko normę ruchu.
5. Dopiero wtedy testować dodatkową wartość SD względem retrieval bez SD i bezpośredniego wyboru danych przez gradienty.
6. Uwzględniać koszt selekcji, generatora i gradientów kandydatów w czasie do docelowego AP.


![Audyt historycznego celu K1000](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/AUDYT_OSIAGALNOSCI.png)

Wykres: maksima po 600 tokenach w każdym kanale kontra konieczna granica oraz rozkład energii delty. Dotyczy jednego historycznego celu J128. Źródło: własny audyt CPU i jego zapisane hashe. [Wersja do eksportu PDF](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/AUDYT_OSIAGALNOSCI.pdf).

### Etykiety epistemiczne

- **[REPO]**: odczytane istniejące pliki; wyniki historyczne nie były uruchamiane ponownie.
- **[CPU]**: nowe pomiary w tym researchu, wyłącznie CPU.
- **[LIT]**: treść pełnych tekstów; cytowania prowadzą do bibliografii w BIBLIOGRAFIA.md.
- **[WNIOSEK]**: własna dedukcja lub interpretacja; nie twierdzenie zapożyczone z pracy.
- **[HIPOTEZA]**: przewidywanie do sprawdzenia.
- **[TEST PROPONOWANY]**: test nie został wykonany w tym researchu.

## 2. Co faktycznie sprawdzono

Użyto jawnie wskazanych umiejętności brainstorming i paper-search. Klasyfikacja: autoryzowany research/spike, którego wynikiem jest raport. Nie wymagano ponownej zgody na już zlecone pytanie i metodę. Nie tworzono kodu produktu ani treningów GPU.

Undermind: get_orientation, list_workspaces, kontrola wcześniejszych wyszukiwań, launch_deep_search, inspect_deep_searches, lookup/get_paper_info, read_pdfs i pobieranie dostępnych PDF. Nowe wyszukiwanie zakończyło się z 166 pozycjami; przejrzano metadane pierwszych 50, a pełne teksty dobrano według pokrycia krytycznych pytań. To celowany przegląd, nie systematyczny review wszystkich 166 prac ani dowód pierwszeństwa.

[Wyszukiwanie Undermind](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Frozen%20detector%20oracle%20activations%20reachability%20and%20learning%20transfer%20K1000%20October%202026).

paper-search CLI działał i przeprowadził wyszukiwania arXiv/Crossref oraz odczyt pełnego AdamW. Część wyników luźnych zapytań była nieadekwatna; nie traktowano ich jako dowodów. Początkowy odczyt pełnego tekstu zawiódł przez kodowanie konsoli cp1250; powtórzenie z UTF-8 zakończyło się sukcesem. Undermind nie miał PDF AdamW, więc pobrano i odczytano pierwotny arXiv. Jeden lookup zgłosił częściową niedostępność backendu; brak pozycji nie został uznany za brak pracy. Dokumentacja PyTorch zwracała błędy HTTP; użyto oficjalnego kodu źródłowego wersji 2.6.0 i pracy AdamW.

W katalogu literature zapisano 24 odrębne pełne PDF oraz ekstrakty z numerami stron. Szczegóły i hashe są w FULLTEXT_MANIFEST.json. Undermind czytał 23 dostępne prace w sześciu paczkach; AdamW odczytano lokalnie przez CLI i PDF. Analiza obejmowała treść artykułów, nie tylko abstrakty. Zachowano surowe notatki read_pdfs, również ich ograniczenia i pomyłki: np. GELU nie oznacza nieujemności, a wyniki dla tokenów tekstowych nie są wynikami dla tokenów obrazu.

Nowy audit_cpu.py wczytał checkpoint na CPU, zbadał fc2 i norm2/fc1, zsyntetyzował wektory do testu projekcji i odczytał historyczne before/after. Nie wykonał forwardów obrazów ani kroków treningu. Małe migawki dowodów i hashe przechowano w evidence. Nie kopiowano zbiorów obrazów, historii latentów ani innych checkpointów.

## 3. Aparat i pochodzenie K1000

[REPO, CPU] Checkpoint:
assets/MalyViT_K1000_SD_conditioning/MODEL_VIT_K1000.pt.
SHA256: 3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e.
Kontener zawiera net i krok=1000, **nie zawiera optymalizatora**.

[REPO] Pochodzenie w wyznacz.py prowadzi do maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k1000.pt. WYNIKI.md tego biegu deklaruje osobny trening od zera, seed47, 1000 aktualizacji, bazę 1500. Nie mieszać tego checkpointa z siłami innych biegów seed47 ani z K8000. Sam numer kroku i seed nie identyfikują wspólnej trajektorii.

[REPO] Model: łodyga konwolucyjna, 6 bloków timm, szerokość residual 192, mlp_ratio 4. Ostatnia gałąź:
norm2 -> fc1:192→768 -> GELU -> fc2:768→192 -> residual -> końcowy LayerNorm -> Linear 192→192 -> GELU -> Linear 192→5.
Dla RGB320×480 łodyga daje siatkę 20×30, czyli 600 tokenów. 768 to szerokość MLP po GELU, nie szerokość całego residual stream. Liczby te są własnością konkretnej implementacji.

Cząsteczka oznacza reprezentację jednego tokenu. Makrowektory e_j są wspólną bazą; mikrowektory z_j e_j są składowymi danej cząsteczki. Globalny układ jest wspólną przestrzenią współrzędnych, nie średnią tokenów.

[REPO] PLAN.md nowej rodziny oracle ma wiele poprawnych zabezpieczeń: frozen/eval, hash wag i buforów, per-image optimizer, top-1 ranking, oba progi w tej samej iteracji, rzeczywiste kroki po sukcesie, brak paddingu, zapis pierwszego sukcesu, GroupKFold i rozdzielenie in-sample od transferu. To dobre elementy aparatu.

Nie potwierdzono ukończonego wyniku oracle dla 1500: odczytany README deklarował przygotowanie, a WYNIKI.json jest deklaracją rodziny i wymaganych artefaktów, nie tabelą wyników produkcyjnych. Repozytorium jest współdzielone i zmienia się równolegle; raport odnosi się do zapisanych migawek, nie do gwarantowanego bieżącego stanu innych wykonawców.

## 4. Reachable manifold: cztery różne zbiory

[WNIOSEK] Zdefiniuj:

- M_theta = {Z_theta(x): x jest dopuszczalnym obrazem}.
- M_natural = obraz naturalnego rozkładu scen; jest podzbiorem M_theta.
- M_generator = {Z_theta(G_eta(v)): v jest dopuszczalnym wejściem generatora}.
- R_update = {Z_(theta+Delta_theta(B,state))(X): B jest dopuszczalną paczką lub sekwencją paczek}.

Zbiór oracle post-GELU z samą dolną granicą jest znacznie luźniejszą relaksacją. Sukces w nim nie implikuje członkostwa w żadnym z powyższych zbiorów. Osiągalność przez swobodny obraz nie oznacza osiągalności przez konkretny SD; osiągalność przez SD nie oznacza osiągalności wspólną aktualizacją wag dla wielu stałych sond.

### 4.1. Konieczne górne granice: nowy konkretny kontrprzykład

[WNIOSEK matematyczny, CPU] Dla jednego tokenu niech h∈R^d, d=192, będzie wejściem norm2:

$$u=\frac{h-\bar h\mathbf1}{\sqrt{\operatorname{var}(h)+\epsilon}},\qquad
\mathbf1^T u=0,\qquad \|u\|_2^2\le d.$$

LayerNorm daje $\gamma\odot u+\beta$.

Dla wiersza a_j macierzy fc1 i bias b_j:

$$p_j=a_j^T\beta+b_j+(a_j\odot\gamma)^T u.$$

Niech $P=I-\mathbf1\mathbf1^T/d$. Z nierówności Cauchy'ego–Schwarza:

$$|p_j-c_j|\le\sqrt d\,\|P(a_j\odot\gamma)\|_2,
\qquad c_j=a_j^T\beta+b_j.$$

To jest skończony, konieczny przedział pre-GELU wynikający wyłącznie z wag. Po przeprowadzeniu przedziału przez dokładny GELU otrzymujemy konieczny przedział z_j. Minimum GELU wypada blisko -0,75179 na wejściu i wynosi około -0,16997121; górną granicę wyznacza maksimum wartości GELU na końcach przedziału. Samo clamp do globalnego minimum GELU pomija konieczne górne granice.

[CPU] Niezmieniony historyczny before spełnia wszystkie obliczone granice. after narusza 15 661 granic górnych; maksymalny nadmiar ponad właściwą granicę kanału wynosi 42,8913. Minimalny błąd dowolnej inwersji ogranicza:

$$\operatorname{RMSE}(Z_\theta(x),Z_*)\ge
\frac{\|Z_*-\operatorname{clip}_{\mathrm{box}}(Z_*)\|_F}{\sqrt{600\cdot768}}
\approx2{,}50349.$$

Plik celu: outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz,
SHA256:2e00434405969a0742b30f15e16cb6156f99e0a6aae2ee9e0f93ce90249a66df.

**Wniosek ścisły:** ten konkretny pełny target nie ma preimage przy frozen K1000, nawet po dopuszczeniu dowolnego RGB. Niewielkie błędy FP32 nie tłumaczą nadmiaru 42,89. To nie jest twierdzenie o wszystkich przyszłych oracle targets ani o możliwości uzyskania tego samego zachowania głowy innym latentem. Po zmianie wag zmieniają się także granice.

Spełnienie granic jest konieczne, ale niewystarczające. Wektor 768D powstaje jako GELU(fc1(LN(h192))); lokalna ranga mapy do 768 wynosi najwyżej 191 przez niezmienniczość LayerNorm na wspólne przesunięcie. Nie wszystkie dopuszczalne współrzędne można ustawiać niezależnie. Dodatkowo wszystkie tokeny obrazu mają wspólne upstream constraints. Nie należy opierać argumentu wyłącznie na wymiarze pikseli: tutaj liczba skalarów RGB320×480 wynosi 460800, tyle samo co całego latentu. Silniejsze ograniczenie wynika z konkretnej architektury i normalizacji.

[LIT] Mahendran/Vedaldi [R1](https://doi.org/10.1109/CVPR.2015.7299155), PDF s.2–3, równania 1–3, formalizują inwersję z prior i błąd reprezentacji, a nie gwarancję naturalnego preimage dowolnego wektora. Mishra i in. [R17](https://arxiv.org/abs/2604.09839), PDF s.3–5, §3–4, dowodzą braku surjektywności dla dyskretnych promptów przy własnych założeniach. Ich argument z przeliczalnością promptów nie przenosi się automatycznie na ciągłe RGB. Tutaj korzystamy z osobnego dowodu przez LayerNorm.

### 4.1a. Jak ograniczyć oracle bez udawania pełnej osiągalności

[TEST PROPONOWANY] Porównać trzy parametryzacje: swobodne z768, z ograniczone koniecznym box i normą oraz z=GELU(fc1(norm2(h192))) z optymalizowanym h192. Ostatnia zachowuje lokalną mapę norm2/fc1 i jej sprzężone współrzędne. Nadal nie gwarantuje, że h każdego tokenu może powstać z jednego obrazu ani że jest zgodny z residual. To silniejsza relaksacja, nie dowód członkostwa w naturalnej rozmaitości.

Cel do transferu warto definiować jako zbiór: poprawna geometria, ranking i ograniczone false positives przy minimalnym widocznym ruchu. Wybranie jednego arbitralnego wektora z tego zbioru tworzy niepotrzebny problem inwersji. Kilka startów oracle pozwala zmierzyć, czy podobny wynik detekcji wymaga podobnego kierunku, czy tylko dowolnej reprezentacji równoważnej dla głowy.

### 4.2. Nullspace fc2: co wolno interpretować

[CPU] fc2 ma rangę 192, najmniejsza wartość singularna 0,2920, największa 1,1364; liczba uwarunkowania około 3,8915. Kernel o wymiarze 576 nie jest skutkiem przypadkowej prawie-osobliwości.

[WNIOSEK] Przy stałym residual r:
D_i(z_i;r_i)=head(LN(r_i+W2 z_i+b2)).
Jeśli n∈ker(W2), D_i(z_i+n;r_i)=D_i(z_i;r_i) dokładnie w matematyce rzeczywistej. Pełna siatka ma co najmniej 345 600 niewidocznych kierunków, zanim uwzględnimy dodatkową niejednoznaczność głowy.

Rozkład:
delta = delta_row + delta_null,
delta_row=W2^+W2 delta.
Rzut na rowspace zachowuje W2 delta. Test CPU daje względny błąd 2,88·10^-15. To nie jest jeszcze test całego hook-forward na GPU.

Rzut rowspace można stosować diagnostycznie bez zmiany wyniku tail. Nie wolno uznać go automatycznie za naturalny target: może naruszyć GELU bounds lub upstream manifold, a w historycznym J128 zmienić pozostałe 640 kanałów. Minimalną korektę przy ograniczeniach trzeba rozwiązać ponownie.

Gradient czystej straty detekcji ma postać W2^T g, więc leży w rowspace. Adam skaluje współrzędne osobno; diag-preconditioner nie musi zachowywać tej podprzestrzeni. Projekcja/clamp oraz kara korekty także mogą tworzyć składową null. Syntetyczny przykład na rzeczywistej W2 daje 26,884 procent energii pierwszego kierunku Adam w kernelu. To demonstracja możliwości, nie statystyka przyszłego oracle.

W rzeczywistym historycznym J128 delta ma 74,647 procent energii null. Dlatego MSE pełne 768 może mierzyć głównie coś, co nie ma wpływu na detekcję frozen tail. Po aktualizacji wag W2 jego kernel może się zmienić; niewidoczność dotyczy tego checkpointa i stałego residual. W ocenie uczenia potrzebna jest projekcja dla właściwego stanu oraz rzeczywisty efekt zadaniowy, nie uniwersalne odrzucenie tych współrzędnych.

[LIT] Grant i in. [R13](https://arxiv.org/abs/2511.04638), PDF s.6, eq.4, rozróżniają behavioral nullspace od szkodliwego off-manifold. Makelov i in. [R12](https://arxiv.org/abs/2311.17030), §3, pokazują iluzję mieszanego kierunku dormant+disconnected. Nie twierdzą, że czysta edycja dokładnego kernelu fc2 sama zmienia output.

### 4.3. Ograniczenia głowy i końcowego LayerNorm

[WNIOSEK] Final LayerNorm usuwa wspólny shift q+c1 dokładnie. Dodatnia zmiana skali q ma przybliżoną niezmienniczość, ale przy epsilon>0 nie należy nazywać jej dokładną. Lokalny Jacobian pięciu surowych wyjść względem 768 współrzędnych ma rank≤5; jego lokalny kernel ma wymiar≥763. To lokalna nieidentyfikowalność, nie 763 globalnych liniowych kierunków bez efektu przy dużych perturbacjach.

Final LayerNorm i skończone wagi również ograniczają dostępne wartości logitów. Nie zakładać, że dowolne confidence lub ramka są osiągalne. Dla odległości ltrb z exp ramka musi zawierać środek wybranej komórki. Przy bardzo małej GT i niekorzystnej siatce tokenów osiągalny IoU może mieć ograniczenie geometryczne. Należy zbadać geometrię i head-constrained maximum przed przypisaniem porażki optimizerowi.

Niepowodzenie w 500 krokach obala twierdzenie o sukcesie ustalonej procedury w 500 krokach, **nie matematyczne istnienie rozwiązania**. Należy raportować budget_exhausted, a nie unreachable, chyba że istnieje niezależny dowód ograniczenia.

## 5. Oracle confidence i ranking: mocna interwencja, słaby dowód semantyki

[REPO] PLAN uwzględnia ranking i tłumienie innych komórek. To istotna poprawka wobec samego podniesienia logitu target token.

[WNIOSEK] Oracle ma dostęp do dokładnej GT i optymalizuje odpowiedź głowy. W jednym obrazie rozwiązanie może być bliskie zaprogramowaniu lokalnego output. Gdy za interwencją nie ma token mixing, ruch tokenów konkurencyjnych może być globalnie podobny dlatego, że wszystkie mają tłumić confidence. Średnia delta może stać się bias mapy pozycji, a nie korektą obrazu.

Należy rozdzielić:

- poprawę geometrii tej samej komórki;
- zmianę komórki top-1;
- podniesienie confidence właściwej ramki;
- obniżenie confidence złych ramek;
- zmianę liczby false positives, recall i precision;
- kalibrację na nowych obrazach.

Top-1 confidence≥.90 i IoU≥.90 na training oracle nie gwarantuje dobrego AP50: może pozostać wiele innych wysoko punktowanych błędnych detekcji. Optymalizacja progu confidence może pogorszyć calibration. GT jest użyta do konstruowania celu, więc zgodność z nią jest częściowo założeniem zadania.

[LIT] Nguyen i in. [R2](https://doi.org/10.1109/CVPR.2015.7298640), PDF s.1,4,7, pokazują bardzo pewne klasyfikacje nierozpoznawalnych obrazów. To klasyfikacja, nie bezpośredni eksperyment tego detektora. Cao/Johnson [R11](https://arxiv.org/abs/2106.13933), §3, eq.1–9, optymalizują obrazy dla zadanego layout i rozdzielają classification/regression cues (§4.3). Ich transfer rozpoznawania przez inne detektory nie jest dowodem korzyści treningowej.

[TEST PROPONOWANY] Oprócz obu progów zapisywać mapę 600 confidence, ranking, decoded top 10, local-max suppression, assignment do GT, IoU tej samej komórki i całego top-1. Oceniać również null/no-tool oraz obrazy z wieloma obiektami, jeśli są dopuszczalne w docelowej domenie. Nie tworzyć eksperymentu wyłącznie na 1500 przykładach po jednej GT, a następnie twierdzić, że rozwiązano ogólne false positives.

## 6. Średnia delta i korektor z kontekstem

### 6.1. Co może działać

[HIPOTEZA] Jeśli K1000 ma wspólny błąd lokalizacji/logitów lub systematyczny bias dla określonego kontekstu, jego korekty mogą mieć powtarzalny składnik. Średnia per spatial token jest tanią kontrolą; conditional corrector może modelować ruch zależny od z0, pozycji i otoczenia. Nie trzeba z góry odrzucać tego wariantu.

[LIT] LabelEnc [R8](https://doi.org/10.1007/978-3-030-58595-2_32), PDF s.5–8, równania 7–9 na s.6, pokazuje udany transfer label-derived desired features do **rzeczywistego uczenia detektora**, z pomocniczą stratą i zgodną głową. To najbliższy pozytywny argument dla celu pośredniego. Metoda ma dwa etapy i nie jest dowodem, że niezależna edycja frozen MLP daje użyteczny cel.

### 6.2. Gdzie średnia zawodzi

[WNIOSEK] Mu_j=E_i[delta_ij] jest estymatorem średniego wektora, nie wektora maksymalizującego detekcję nowego obrazu:
D(z+E delta) ≠ E D(z+delta).
Przeciwne ruchy znoszą się; niesymetryczny rozkład powoduje dominację obserwacji odstających. Ten sam indeks j oznacza pozycję obrazu, a nie zawsze ten sam fragment narzędzia. Średnie tłumienie tła może pomagać na rozkładzie z obiektem w jednej pozycji, a szkodzić po przesunięciu lub zmianie kamery.

Pomiar cancellation:
C_j=||sum_i delta_ij||/(sum_i||delta_ij||).
C_j≈0 wskazuje brak wspólnego kierunku, ale sam nie rozstrzyga o conditional structure. Liczyć osobno rowspace i nullspace, magnitudę, obszar narzędzia/tła i filmy. Robust median/trimmed mean oraz shrinkage do zera powinny być kontrolami; mean może być właściwy przy quadratic risk, ale head risk jest inny.

Uśrednianie delty z różnych aktywnych ograniczeń GELU nie zachowuje ograniczeń na nowym z0. Clamp po dodaniu zmienia estymowaną korektę; wymagany jest raport obu wielkości.

### 6.3. Conditional corrector i brak leakage

Reżim może być odkryty opisowo z danych (z0, delta, trajectory). W deploy przypisanie ma zależeć tylko od naturalnego z0 i dostępnego kontekstu. Użycie oracle delta, GT, czasu pierwszego sukcesu lub gradientu względem prawdziwej GT w gating na nowym obrazie jest leakage.

W każdym podziale walidacji:

1. skalowanie, dobór cech, klasteryzacja, liczba klastrów i korekty dopasowane tylko na filmach train;
2. model przypisania reżimu uczony z naturalnego z0 i kontekstu, bez celu oracle z danych odłożonych;
3. wszystkie tokeny i klatki tego samego filmu pozostają razem;
4. dobór hiperparametrów na osobnej walidacji; końcowy test nietknięty;
5. podstawową miarą jest użyteczność głowy i generalizacja, nie zgodność z klastrem oracle.

**Istotne:** GroupKFold w obrębie 1500 obrazów użytych wcześniej przez K1000 sprawdza transfer **korektora**, lecz nie daje niezależnego testu samego detektora. Do końcowego twierdzenia potrzebne są świeże filmy nieużyte w treningu K1000, generatora, targetów ani selekcji.

Porównać średnią 600×768, średnią globalną, ridge/MLP/adapter z kontekstem, korektę z reżimami i shrinkage, brak korekty oraz oracle per obraz. Regresja bez klastrów sprawdza, czy dyskretyzacja reżimów wnosi dodatkową wartość.

## 7. Trajektorie i reżimy: artefakty optimizerów

[REPO] Wspólny schedule oracle: 1–100 LR 0,05; 101–250: 0,15; 251–400: 0,05; 401–500: 0,01. Po pierwszym sukcesie zmienia się cel na minimalną korektę z barierami. Każdy obraz ma własną chwilę przełączenia.

[WNIOSEK] Wspólny numer iteracji jest prawidłowym zegarem proceduralnym, ale nie semantycznym. Cząsteczki mogą grupować się według zmian LR, aktywnych granic clamp, pierwszego sukcesu i zmiany straty. Etykieta reżimu może przewidywać fazę optimizera albo początkową trudność obrazu.

Po sukcesie dalszy ruch jest prawdziwy, ale dotyczy innego celu. Brak paddingu usuwa jedną pułapkę; nie usuwa nieliniowej selekcji faz. after jest stanem minimalnej normy spośród odwiedzonych sukcesów, więc jego krok jest zależny od obrazu. Średnia delta after nie jest średnią stanu w tej samej iteracji.

[TEST PROPONOWANY] Na małym panelu:

- ten sam cel i różne schedule oraz Adam/SGD;
- stały schedule, różne początkowe perturbacje;
- cele z permutowanymi GT jako kontrola task specificity;
- stały cel bez przełączenia versus switch;
- kontrolowane przesunięcie/kolor/geometria z pozostawieniem innych cech;
- klasteryzacja widocznego ruchu W2 delta kontra pełne 768D;
- stabilność klastrów między seedami i filmami;
- predykcja z samego k, LR, fazy celu, normy i pozycji celu jako kontrola.

Reżim semantyczny powinien przewidywać niezależną cechę lub użyteczność na held-out filmach i przetrwać zmianę schedule. Nie wystarczy atrakcyjna animacja, silhouette lub dobra rekonstrukcja DMD.

[LIT] Geirhos i in. [R14](https://arxiv.org/abs/2306.04719), PDF s.6, Fig.5, pokazują, że visualizations mogą mieć odmienne wewnętrzne przetwarzanie niż naturalne obrazy. To uzasadnia kontrolę naturalnych odpowiedników; nie dowodzi automatycznie, że każdy cel SD jest artefaktem.

## 8. Closure AdamW, Koopman/DMDc i Langevin

### 8.1. Dwa różne systemy

Oracle:
z_(k+1)=Pi_C(z_k+Adam(g_z(z_k,GT,phase_k),m_k,v_k,lr_k)).
Stałe wagi, zmieniane aktywacje.

Uczenie:
theta_(t+1)=AdamW(theta_t,g_theta(B_t),m_t,v_t,lr_t),
z_(t+1)=Z_(theta_(t+1),BN_(t+1))(x_fixed).
Stałe piksele sondy, zmieniane wspólne wagi i bufory.

**Nie są tym samym układem dynamicznym.** Dopasowany operator oracle nie jest prognozą uczenia. Duża zgodność stanu i gradientu przy oracle może po prostu odtwarzać znaną formułę Adam.

Pełny stan uczenia obejmuje wagi, momenty AdamW, liczniki kroków, bufory BN, scheduler, sampler/RNG i augmentację. Sam z, jego velocity i mean nie muszą być Markovian: dwa stany modelu dające to samo z mogą mieć różne gradienty i przyszłość.

Dla oracle pełna znajomość z, z0, GT, stałego kontekstu końcówki, m, v, k i stanu przełączenia celu wystarcza do deterministycznego następnego kroku z konkretnymi ustawieniami numerycznymi. Wtedy podstawowym baseline jest **dokładny optimizer update**, nie persistence. Dopasowanie DMDc do gradientu/momentów nie może być samo przedstawiane jako odkryte prawo.

### 8.2. Co DMDc może wnieść

[LIT] Proctor/Brunton/Kutz [R3](https://doi.org/10.1137/15M1013857), PDF s.3–4, eq.14,25–34: x'=Ax+Bu jest modelem warunkowym ze zmierzonymi wejściami. Oddzielenie A i B wymaga odpowiedniej informacji/rangi macierzy [X;U]. Jeśli u=Kx w danych, sam closed-loop może nie identyfikować osobno A i B bez dodatkowych warunków. To ograniczenie identyfikacji, nie dowód bezużyteczności DMDc.

[WNIOSEK] Gradient celu jest funkcją stanu. Wrzucenie z, gradientów i momentów tworzy correlated regressors. Należy sprawdzić effective rank, conditioning i excitation oraz porównać z mechanistyczną formułą optimizerów. Nie wprowadzać u_t=delta_(t→t+1) jako wejścia prognozy tego samego kroku.

Warunki:

- podziały po obrazach/filmach oraz chronologii;
- skalowanie/POD tylko train;
- stan i wejście w tej samej fazie celu;
- input znany przed zmianą;
- H>1: jawnie oznaczyć recorded-future-input oracle albo przewidywać/recompute przyszłe gradienty;
- measured next input nie jest znanym samym schedule paczek.

[WNIOSEK] Skończony model Koopmana wymaga, aby wybrane obserwable miały domkniętą ewolucję w rozpatrywanym zakresie; duża zgodność regresji tego nie dowodzi. Praktyczny test niedomknięcia: znaleźć stany o podobnym z i prędkości, ale różnych m/v, gradientach lub BN, a następnie porównać ich przyszłe przyrosty przy tej samej paczce. Systematycznie różne przyszłości obalają wystarczalność takiego stanu zredukowanego. Dokładać zmienne wtedy, gdy poprawiają prognozę na odłożonych trajektoriach, a nie tylko dopasowanie historii.

### 8.3. Langevin i pamięć

Langevin może być lokalnym modelem opisowym oscylacji, tłumienia i pamięci. Składnik stochastyczny może reprezentować niedomknięcie i losowość paczek, ale nie automatycznie termodynamiczną kąpiel cieplną. Residual nie staje się białym szumem przez nazwanie go noise. Potrzebne są testy autokorelacji, warunkowej wariancji, ogonów rozkładu i zależności od LR.

Współczynniki opóźnień GLE mogą przybliżać pamięć; bez wyprowadzenia w formalizmie projekcji nie nazywać ich odkrytym jądrem Mori-Zwanziga ani temperaturą. Deterministyczne drgania Adam pod stałym target nie są dowodem fizycznego Langevina cząsteczek semantycznych.

[LIT] Liu/Li [R15](https://arxiv.org/abs/2608.19762v1), PDF s.3–7, eq.1–2 i twierdzenie 3.1: state=(theta,m,v), lokalny signed response i propagacja poprzez przyszłe Jacobiany. Ta sama przyszła paczka nie daje tych samych przyszłych gradientów po rozjechaniu stanów. Praca jest preprintem z 2026 roku, bez potwierdzonej recenzowanej publikacji. Deng i in. [R16](https://arxiv.org/abs/2605.18814v1), §3–5, AppendixB, pokazują znaczenie optimizer mismatch i narastania błędu Taylor. Ich wyniki nie zastępują testu chirurgicznego SmallViT.

## 9. Interpretacja 95–97 procent i 98,5 procent

[REPO] maly_vit_odpowiedz_latentu_na_paczke/WYNIKI.md definiuje:
skill = 1-SSE_model/SSE_no_change.
Na H1 model z bieżącym pchnięciem daje 0,954/0,956. To redukcja błędu względem persistence dla 180 sond × 768 po poolingu tokenów regionu narzędzia. Nie oznacza to procentu poprawnych predykcji, AP ani dokładności ruchu wszystkich 900 000 cząsteczek ani korzyści doboru danych.

0,985 na H10 wymaga pełnych sił kolejnych kroków policzonych w prawdziwych stanach. Znajomość samego schedule nie daje momentów i gradientów w tych stanach. Z tej samej historii wynika spadek do około 0,63 przy modelu samych pchnięć i stałych współczynnikach, a uczciwe H10 z samym bieżącym wejściem daje około 0,276/0,282.

[REPO] Nowa pełniejsza analiza DMDc na 3 sondach × 600×768:

- DMDc_delay H1/H5/H10: skill 0,3383/0,2060/0,0499.
- prosty force_inertia:0,9729/0,8684/0,5892.
- dla tokenów narzędzia DMDc_delay:0,2361/0,1070/-0,0474.
- 64 mody POD wyjaśniają 98,98 procent wariancji train states, ale oracle projection test displacement H1 ma skill≈0,5003.

Wyjaśnienie wariancji stanu nie jest wyjaśnieniem wariancji ruchu. Duża struktura statyczna może dominować stan; przyszłe przyrosty wykraczają poza wybraną POD. Wynik 97,29 procent H1 force_inertia pochodzi z pełnych tokenów 3 sond i prawdziwego bieżącego wymuszenia, więc nie utożsamiać go z wcześniejszym wynikiem 95,4 dla reprezentacji po poolingu. H5/H10 tego raportu nadal używają recorded future forces.

SSE musi być raportowane razem z baseline SSE, RMSE, cosine displacement, error w visible/head-space i task metrics. Bardzo mały mianownik czyni skill niestabilnym. Niepewność bootstrapować blokami po filmach lub niezależnych trajektoriach; 900 000 tokenów nie jest 900 000 niezależnymi próbami.

## 10. Inwersja SD, residual i DINO retrieval

### 10.1. Zgodny checkpoint to warunek konieczny

SD trenowany na Z_K1000 naturalnych obrazów nie ma automatycznie zgodnego conditioning dla Z_K8000. Ta sama liczba 768 i indeks kanału nie oznaczają tej samej semantyki. Trzymać hash wag, hook, preprocessing, kolejność siatki i dtype. Jeżeli obecny student zmienił wagi, target odniesienia musi być zdefiniowany na jego aktualnej mapie albo jako zachowanie lub odniesienie do nauczyciela, nie ślepo porównywany w surowych współrzędnych.

### 10.2. Dlaczego samo z nie wystarcza

[WNIOSEK] Przy oracle oryginalnego obrazu:
q_target = q0 + W2(Z_target-Z0),
gdzie q0 zawiera stary residual. Nowy obraz x_gen daje:
q_gen = r(x_gen)+W2 Z(x_gen)+b2.
Nawet Z(x_gen)=Z_target nie wymusza r(x_gen)=r0. fast_tail z cache q0 nie jest oceną naturalnej generacji. Potrzebne są naturalny wynik głowy oraz residual/post-block features jako kontrola.

Zniekształcony target może mieć łatwe równoważne zachowanie, mimo niemożliwości dokładnego dopasowania surowych współrzędnych. Funkcyjny cel na stanie po końcowym LayerNorm lub logits/boxes bywa właściwszy. Jednak wyjście docelowe może nadal zostać zrealizowane przez obraz oszukujący model, dlatego potrzebne są niezależne kryteria realizmu i semantyki.

[LIT] Mahendran/Vedaldi [R1](https://doi.org/10.1109/CVPR.2015.7299155) oraz Cao/Johnson [R11](https://arxiv.org/abs/2106.13933) dają problem regularized inversion. DeepInversion [R9](https://doi.org/10.1109/CVPR42600.2020.00874), PDF s.3–4, eq.2–7, dodaje statystyki feature/BN i ocenia student learning. To pozytywny przykład, że inversion może być użyteczna, z kontrolami dystrybucji. DPS [R10](https://arxiv.org/abs/2209.14687), §2–3, approx.likelihood i guidance, nie gwarantuje preimage dla niewykonalnego target; generative prior może prowadzić do kompromisu.

### 10.3. Retrieval nie jest influence

[REPO] vit_dino_faiss_35000: 35 000 wyłącznie TRAIN, DINOv2 global CLS, L2/IndexFlatIP, audyt zgodności z cosine i provenance. To poprawny indeks wyszukiwania pilotażowego; sam plan zaznacza możliwe kodowanie tła.

[WNIOSEK] Sąsiedzi DINO odpowiadają na pytanie o podobieństwo w DINO. Mogą być podobne przez rogówkę, oświetlenie, zoom lub pacjenta, a pozbawione użytecznego kierunku aktualizacji dla końcówki narzędzia. Artefakty syntetycznego zapytania mogą sterować rankingiem. FAISS nie dodaje semantycznej gwarancji; realizuje wyszukiwanie wektorowe.

Sprawdzić reranking po tool ROI/patch features, GT geometry i diversity; nie stosować GT testowych do query/gating. Porównać original-image query, reachable target query, SD query, bezpośredni SmallViT feature query i gradient utility. Deduplikować czasowo/po filmach; top 32 kolejnych klatek nie są 32 niezależnymi źródłami informacji.

[LIT] DINO 2021 [R21](https://doi.org/10.1109/ICCV48922.2021.00951), PDF s.5,7–8, wspiera kNN classification i spatial cues, nie update utility. Lokalnie używany DINOv2 to inny model; nie przypisywać mu automatycznie tabel DINO 2021. Geng i in. [R18](https://arxiv.org/abs/2406.05184), §3–4, Fig.2, pokazują przewagę target real retrieval nad SD na testowanych CLIP classification tasks. Nie jest to uniwersalne twierdzenie o chirurgicznej detekcji, lecz mocny argument za retrieval-only baseline.

### 10.4. Istniejący lokalny kontrdowód transferu

[REPO] Sesja 2026-10-08_155828 i bezpośredni raport compatible_k8000_target:

- oracle zgodnego K8000: IoU 0,05063→0,99720, confidence 0,84576→0,90505;
- RMSE do celu: start 1,083592;kontrola 1,121694; wariant 8: 1,121515;
- cosine efektu (wariant minus kontrola) z kierunkiem celu: 0,0213968;
- zrealizowana część drogi 0,07088 procent;
- 8 podmian: IoU +0,0091, confidence -0,0277, guard AP50 -0,0052.

To jedno porównanie seed/puli i inna rodzina K8000. Nie dowodzi niemożliwości nowego K1000. Pokazuje natomiast, że poprawa jednej geometrii i rzeczywisty ruch latentu nie implikują ani uczenia w stronę celu, ani ogólnej korzyści. Dalszy trening kontrolny sam znacznie pomagał; efekt wariantu trzeba liczyć względem niego.

## 11. Weight transfer, attribution i one-step learnability

### 11.1. Wspólny Jacobian jest właściwym mostem

[WNIOSEK] Dla stałych sond X:
Z(theta+Delta theta,X) ≈ Z(theta,X)+J_theta Delta theta.
Niezależny oracle per obraz wybiera osobne delta_i, ale rzeczywiste uczenie ma jedną wspólną Delta theta. Należy sprawdzić:
min_Delta ||J_theta Delta-delta_target||² +lambda||Delta||²,
albo analogiczny problem w visible/task space.

Nie materializować pełnego Jacobian 900000×768×P. Wystarczą JVP/VJP i iteracyjny least squares na reprezentatywnym panelu. Zwykły duży raw target wymaga trust region; dla małych kroków sprawdzić względny błąd Taylora i kierunek. Local projection residual jest ograniczeniem jednego kroku, nie dowodem globalnej niemożliwości wielokrokowej.

**Istotne:** parametry za hookiem, np. fc2 i głowa, mają J_Z=0. Mogą polepszać detekcję bez ruchu mierzonego z. Odwrotnie, ruch z może zostać skompensowany aktualizacjami głowy. Surowa odległość w latencie nie jest kompletnym celem uczenia. Potrzebne są strata detekcji, AP na danych odłożonych i ograniczenie collateral damage.

### 11.2. AdamW candidate update

Dla stanu przed krokiem t i gradientu pełnej paczki g_B po zadeklarowanym clipping:

$$m^+=\beta_1m+(1-\beta_1)g_B,\qquad
v^+=\beta_2v+(1-\beta_2)g_B^2,$$
$$\widehat m=\frac{m^+}{1-\beta_1^s},\qquad
\widehat v=\frac{v^+}{1-\beta_2^s},$$
$$\Delta\theta_B=-\eta\frac{\widehat m}{\sqrt{\widehat v}+\epsilon}-\eta\lambda\theta.$$

Wzór dotyczy standardowego wariantu AdamW bez AMSGrad i bez maximize; wszystkie operacje na momentach są współrzędne.
Liczniki s i grupy parametrów muszą odpowiadać właściwemu optimizerowi; brak gradientu nie musi być traktowany jak gradient zero.

K1000 asset zawiera wyłącznie net,krok. Do autentycznej kontynuacji potrzebny zgodny checkpoint z opt i BN/RNG. Wyzerowany AdamW jest nowym protokołem; można go testować, ale nie przypisywać wyników kontynuacji historycznej.

Predykcja shift=J_Z Delta theta_B; finite-step probe to Z(theta+Delta)-Z(theta). To dwa różne pomiary. S_nowa ze starego raportu jest finite-step, nie JVP. Oddzielić stary i zaktualizowany BN, a wynik końcowy oceniać w takim trybie jak deployment. W pełnym trainerze aktualizacja buforów BN wnosi dodatkowy składnik do zmiany z; sam J_theta Delta theta go nie obejmuje. Należy zamrozić BN w kontroli albo osobno uwzględnić zmianę buforów, a potem zmierzyć cały krok.

Zaproponowany score:
Q_B = ||e||² - ||e-J_Z Delta theta_B||²
     = 2 e^T J_Z Delta theta_B - ||J_Z Delta theta_B||².
Najlepiej e w task-visible space, plus kontrola straty walidacyjnej. Iloczyn skalarny z desired delta bez normy może preferować overshoot. Gradient względem real validation loss daje bardziej bezpośredni cel:
estimated DeltaL_val ≈ gradL_val^T Delta theta_B.
To [WNIOSEK] dla lokalnego kroku, nie gwarancja późniejszego AP.

Per-example gradients nie dodają się dokładnie do rzeczywistych per-example AdamW updates. Mianownik zależy od pełnej paczki i momentów; clipping i train BN też wprowadzają interactions. Można sumować gradients przy fixed independent loss i BN, ale pełny update trzeba przeliczyć dla paczki.

[LIT] AdamW [R4](https://arxiv.org/abs/1711.05101v3), PDF s.3, Algorithm 2, oraz oficjalny kod PyTorch 2.6.0 opisują momenty i bias correction. TracIn [R5](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html), PDF s.3–4, §3.2/eq.1, wyraźnie wymaga modyfikacji formuły aktualizacji dla Adam. Bae i in. [R19](https://arxiv.org/abs/2209.05364), PDF s.6, równanie 8, pokazują, że influence i retraining mogą odpowiadać na różne kontrfaktyczne pytania. Koh i in. [R20](https://arxiv.org/abs/1905.13289) dotyczą grup w convex models; nie stanowią gwarancji nieaddytywnej dynamiki SmallViT.

### 11.3. Jeden krok nie oznacza trwałej korzyści

[TEST PROPONOWANY] Paired rollouts ze wspólnego pełnego stanu:wariant kontrolny i kandydat różnią się tylko bieżącą paczką; następne paczki/RNG są wspólne. Po rozjechaniu wag gradienty są ponownie liczone w prawdziwych własnych stanach. Raportować 1/5/10/25/50 kroków, target-visible movement, stratę walidacyjną i AP. Ranking zapisać przed rollout.

Jeżeli effect zmienia znak albo zanika, score H1 może sterować chwilową fazą drgań. Lokalna historia sugeruje właśnie transient effect; nie wyciągać tego w nieskończoność bez testu paired causal.

[LIT] GLISTER [R6](https://doi.org/10.1609/aaai.v35i9.16988), PDF s.2–5, eq.1–5, wybiera dane według celu walidacyjnego i przybliżenia jednego kroku. GRAD-MATCH [R22](https://arxiv.org/abs/2103.00123) dopasowuje gradient ważonego podzbioru; jego gwarancje dla SGD nie są gwarancjami AdamW. Zhao [R7](https://arxiv.org/abs/2006.05929) dopasowuje gradients, Cazenavette [R23](https://doi.org/10.1109/CVPR52688.2022.01045) rzeczywiste trajektorie uczenia parametrów, a nie sztuczne tory stanów ukrytych. Balles i in. [R24](https://arxiv.org/abs/2312.05021), Fig.1/3, pokazują, że lepszy gradient proxy nie musi wygrywać z random pod względem trafności na teście. To argument za kosztowo uczciwą kontrolą, nie odrzucenie każdej selekcji.

## 12. Minimalny program falsyfikacji przed kosztownymi obliczeniami

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

## 13. Pamięć, kontrakt zapisu i równoległe zmiany

[CPU] Jeden pełny stan 1500×600×768 FP32 zajmuje 2 764 800 000 bajtów. Zapis wszystkich 501 stanów 0–500 wymaga 1 385 164 800 000 bajtów, czyli około 1,385 TB, bez gradientów i momentów. To wyliczenie scenariusza gęstego, nie stwierdzenie, że taki zapis wykonano.

[REPO, pierwsza migawka] Pierwotnie odczytany plan przewidywał before/after/delta oraz trzy pełne milestone, surowo 16 588 800 000 bajtów. Gęsty panel 256 cząsteczek, 501 stanów, 768 osi i czterech pól FP32 to dodatkowe 1 576 009 728 bajtów przy jednakowej długości wszystkich pól. Panel musi oznaczać 256 cząsteczek łącznie; 256 per obraz zwiększyłoby ten składnik do około 2,364 TB. Pełne rzadkie stany nie umożliwiają identyfikacji gęstej dynamiki wszystkich 900 000 tokenów.

Pierwsza migawka wykazała także niezgodność: PLAN wymagał FP32, a writer pelne_latenty.py:102 i test:80 używały float16. Dowody tej wersji pozostają zachowane; nie należy traktować ich jako gwarantowanego bieżącego stanu repozytorium.

**[REPO, końcowy ponowny odczyt] Równoległy wykonawca zmienił plan i kod podczas researchu.** Writer używa teraz float32, również przy zbieraniu milestone. Bieżący plan zachowuje pełne kroki 100 i 500; krok 0 jest before bez duplikacji. Panel to 16 obrazów × 8 tokenów, czyli 128 cząsteczek łącznie. Runner przekazuje tokeny z panelu przypisanego do indeksu obrazu, więc nie jest to 128 per każdy z 1500 obrazów. Zapis z obejmuje 501 stanów, a gradient/m/v po 500 stanów przed aktualizacją. Rzeczywisty surowy budżet tak zadeklarowanych tensorów wynosi:

- before/after/delta: 8 294 400 000 bajtów;
- dwa milestone: 5 529 600 000 bajtów;
- panel 128×768×4×(501+3×500): 786 825 216 bajtów;
- razem: 14 610 825 216 bajtów, około 14,611 GB przed małą telemetrią i narzutem plików.

Bieżący plan deklaruje wstrzymanie deploy i dużych zapisów do zakończenia audytu miejsca. Ten research nie łączył się z Edenem, nie sprawdzał aktualnego df i nie potwierdza wartości wolnego miejsca z planu. Wymagane jest rzeczywiste ponowne sprawdzenie przed uruchomieniem.

Końcowe kopie i hashe są w evidence/final-read/MANIFEST.json. Nie uruchamiano cudzych testów ani ich produkcji. Korekta dtype w writerze jest odczytanym faktem, nie potwierdzeniem całego kontraktu zapisu. Przy wznowieniach trzeba sprawdzać dtype, identyfikatory stanów i zgodność momentów z krokiem.

Wszystkie twierdzenia o częstotliwościach i pamięci muszą dotyczyć gęstego panelu. Obecne pełne stany 0/100/500 są jeszcze rzadsze niż wcześniejsze 0/25/100/500. Próbkowanie panelu powinno pokrywać filmy, narzędzie/tło, trudność i wielkość ruchu. Nie sklejać niezależnych trajektorii w fałszywe przejścia.

## 14. Co zmienić w pipeline

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

## 15. Zakres zakończenia i ograniczenia

Zakończono analizę, pełnotekstowy przegląd kluczowych prac, audyt CPU checkpointa, archiwizację dowodów oraz opracowanie falsyfikowalnych hipotez i testów. Walidacja empiryczna całego łańcucha pozostaje do wykonania; proponowane forwardy, treningi, kontynuacje i generacja SD nie zostały uruchomione.

Zapisy wykonano wyłącznie w nowych katalogach tego researchu. Cudze eksperymenty pozostały nietknięte; nie uruchomiono GPU ani jobów, nie wykonano commita. Repozytorium było współdzielone i zawierało istniejące zmiany. Późniejsze wyniki i poprawki innych wykonawców należy porównywać z hashami zachowanych migawek.

Dokumenty pomocnicze:

- [BIBLIOGRAFIA.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/BIBLIOGRAFIA.md):24 prace, DOI/primaryURL, strony/sekcje i zakres dowodu.
- [CLAIMS_EVIDENCE_FAILTEST.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/CLAIMS_EVIDENCE_FAILTEST.md): mapowanie teza→dowód→test obalenia.
- [REKOMENDACJE.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/REKOMENDACJE.md): krótka kolejność decyzji przed compute.
- evidence/CPU_AUDIT.json i NATURAL_GELU_BOUNDS.npz: liczby i kanałowe ograniczenia.
- literature/FULLTEXT_MANIFEST.json: pełne teksty i hashe.
