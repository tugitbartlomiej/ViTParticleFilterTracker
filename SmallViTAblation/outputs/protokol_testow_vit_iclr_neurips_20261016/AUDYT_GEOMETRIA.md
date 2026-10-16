# Protokół G01–G28: instrument, geometria i przyczynowe interwencje w SmallViT

Data: 2026-10-16. Status: **specyfikacja przyszłych testów**, bez nowych obliczeń modelu, treningów i zmian aplikacji. Jest to szczegółowy załącznik do katalogu głównego przygotowywanego w tym samym katalogu. Budżet i podziały katalogu głównego są nadrzędne. Testy współdzielą przebiegi, zapisane punkty i operatory: 28 pozycji nie oznacza 28 niezależnych dużych eksperymentów.

Podstawa lokalna: `experiments/plan_publikacji_neurips_20261016/PLAN.md`, `outputs/plan_publikacji_neurips_20261016/audyt_matematyczny.md`, wcześniejszy `outputs/research_fizyczny_mikroskop_20261015/math_audit.md`; kod `experiments/maly_vit_od_zera_35000_21epok_20261009/kod/{model,trening}.py`, `app/{pole_podkrokow,pulapka_obliczenia,ogniskowa}.py` i `experiments/dolina_odpowiedzi_t226_20261015/policz.py`.

## A. Kontrakt badania i matematyka wspólna

**Pytanie potwierdzające:** czy wyuczona orientacja lokalnej deformacji/obrotu przewiduje i przyczynowo zmienia odpowiedź detektora ponad zwykłą czułość, normę poprawki i dobrane losowe kontrole? Istnienie S/A, miejscowego mieszania kierunków lub neutralnej poziomicy głowy jest własnością matematyczną; nie jest samo dowodem mechanizmu poprawiającego detekcję.

**Definicje.** Jeden token residualu: x∈R^192; ukryte aktywacje MLP: z∈R^768; pełny obraz: X∈R^(600×192), czyli 115200 współrzędnych. Wszystkie testy głębokości mają zamrożony checkpoint, obraz bazowy i tryb eval. Sześć bloków to 12 różnych map Ψ_k(X)=X+F_k(X), nie dwanaście iteracji jednego pola. Checkpoint t oznacza inny zegar: x_t=f(I;θ_t,b_t), gdzie b_t obejmuje BN. Przekroje GUI są pomocniczymi wizualizacjami, a nie definicją pełnego układu.

**Dokładne MLP**, przy rzeczywistych LayerScale/dropout zidentyfikowanych w G01; dla obecnego wariantu eval i Identity LayerScale:

\[
a=W_1L(x)+b_1,\quad z=\phi(a),\quad F(x)=\sum_{c=1}^{768}z_cv_c+b_2,
\quad J=DF=W_2\operatorname{diag}(\phi'(a))W_1DL.
\]

Przy Q=I−11ᵀ/d, r=Qx, s²=rᵀr/d+ε, Γ=diag(γ):

\[
DL=\Gamma\left(Q/s-rr^\top/(ds^3)\right),\quad DL\,1=0,
\quad DL\,r=\Gamma r\,\epsilon/s^3.
\]

Nie usuwać radialnego kierunku jako dokładnej symetrii przy ε>0. Nie zastępować podpisanych z wartościami |z|. Bias b₂ ma znaczenie dla F i jego zer, choć nie występuje w J.

**Dokładny iloraz LN.** U∈R^(192×191) jest ustaloną ortonormalną bazą 1⊥, UUᵀ=Q. J̄=UᵀJU, S̄=(J̄+J̄ᵀ)/2, Ā=(J̄−J̄ᵀ)/2. Dla całego obrazu stosować U_N=I_N⊗U, bez budowania go jawnie. Dowolne g=(c_j1)_j spełnia F_k(X+g)=F_k(X), Ψ_k(X+g)=Ψ_k(X)+g; końcowa głowa usuwa g przez finalną LN. Zatem pełna mapa posiada 600 dokładnych neutralnych kierunków. Odejmowanie średniej po tokenach nie usuwa tego samego gauge.

**Skończony residual.** M=I+J, M̄=I_191+J̄. Dla perturbacji małej:

\[
\Psi(x+\delta)-\Psi(x)=M\delta+R(\delta),\quad
\|M\delta\|^2-\|\delta\|^2=\delta^\top(2S+J^\top J)\delta.
\]

Jeżeli J jest L_J-Lipschitz na odcinku, ||R(δ)||≤L_J||δ||²/2. Wówczas błąd wzoru na zmianę kwadratu normy jest ograniczony przez ||M||L_J||δ||³+L_J²||δ||⁴/4. Empiryczne oszacowanie L_J z sond nie jest certyfikatem globalnego ograniczenia.

**Transport.** Φ_{b←a}=M_{b−1}⋯M_a, z pochodnymi na właściwych stanach naturalnego przebiegu. Dla r ortonormalnych kierunków E: C_E=EᵀΦᵀΦE; singular values ΦE opisują rozciąganie badanego r-wymiarowego elementu, a sqrt(det C_E) jego objętość. To nie pełna objętość 115200D. B=DH w końcowym punkcie, zaś lokalna końcowa odpowiedź to BΦE. Gdy interesuje nas wcześniejsza interwencja, B musi obejmować wszystkie pozostałe bloki, jeśli Φ ich nie obejmuje: nie liczyć ich dwa razy i nie pomijać.

**Główny endpoint.** W parze obraz–perturbacja–seed raportować różnicę znormalizowanego błędu krawędzi ramki między interwencją ukierunkowaną a normmatched sham. Dokładny normalizator, reguła dopasowania detekcji, brak detekcji i kierunek hipotezy po pilocie pochodzą z protokołu głównego i zostają zamrożone przed TEST. Przy transformacji geometrycznej zmienia się GT. Zmiana bbox względem czystego przewidywania nie zastępuje błędu względem właściwego GT. Pomocniczo: logit, margines kandydatów, IoU, fixed-token oraz wynik rzeczywistego dekodera.

**Jednostki.** T = token×podkrok×checkpoint×obraz, X = pełny obraz×podkrok×checkpoint, P = sparowany obraz bazowy i perturbacja przy danym seedzie. Są to rekordy techniczne, nie niezależne replikacje. Niepewność efektów funkcjonalnych klastrować po filmie/zabiegu i seedzie. Do audytu algebraicznego wystarcza błąd numeryczny i kompletność kontroli; nie potrzeba p-value.

**Zasoby.** F oznacza forward potrzebnego fragmentu, D jedno JVP/VJP, J_M pełne 192×192 lokalne MLP J. Nie podajemy minut GPU bez benchmarku. Stan 115200×115200 zajmuje 53,1 GB float32 bez autograd: nie materializować. Główne badanie wykorzystuje s47 jako discovery oraz budżet dwóch dodatkowych seedów rozwojowych i trzech potwierdzających z protokołu głównego. Krótkie stare biegi nie zastępują automatycznie tych replik. Każda dodatkowa siatka jest wariantem, a nie obowiązkiem.

## B. Źródła pierwotne i rola literatury

Zastosowano skill [paper-search](C:/Users/bartl/.agents/skills/paper-search/SKILL.md): pięć zapytań CLI do arXiv/Crossref, następnie weryfikacja trafień na stronach autorów/wydawców. Wyszukiwanie zwracało także nietrafne pozycje; nie liczono ich jako przeczytanych prac. Nie pobierano plików PDF do projektu. Poniższe wyprowadzenia dla konkretnego SmallViT opierają się także bezpośrednio na kodzie i algebrze; cytowanie nie zastępuje G01–G07.

| Ref. | Praca, autorzy i rok | Zweryfikowany zakres i zastosowanie |
|---|---|---|
| P1 | [Layer Normalization — Ba, Kiros, Hinton, 2016](https://arxiv.org/abs/1607.06450) | Odczyt tekstu PDF, definicja normalizacji i statystyk wewnątrz przykładu. G03/G06. Konkretne ε i pochodną z ε weryfikuje kod, nie uproszczony wzór artykułu. |
| P2 | [On Layer Normalization in the Transformer Architecture — Xiong i in., ICML 2020](https://proceedings.mlr.press/v119/xiong20b.html) | Tekst PDF: rozróżnienie Pre-LN/Post-LN i położenie końcowej LN. Motywuje kontrolę rzeczywistego grafu. Nie przenosić ich wyników inicjalizacji wprost na wytrenowane pole. |
| P3 | [How Smooth Is Attention? — Castin, Ablin, Peyré, ICML 2024](https://proceedings.mlr.press/v235/castin24a.html) | Zweryfikowane metadane i abstrakt wydawcy: lokalna czułość attention, zależność od długości sekwencji i normalizacji. Uzasadnia baseline czułości. Pełny PDF z serwera wydawcy nie został odczytany przez narzędzie w tej sesji; nie przypisuję mu nowych szczegółowych twierdzeń. |
| P4 | [Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians — Tomihari, Karakida, NeurIPS 2025](https://arxiv.org/html/2505.19458v4) | Odczyt sekcji 3–7 i ograniczeń; Crossref potwierdza DOI [10.52202/085713-4675](https://doi.org/10.52202/085713-4675). Analiza Jacobianów, normalizacji i dynamiki ma bliski precedens. Model rekurencyjny bez MLP/positional encoding/masking nie jest tożsamy z sześcioma różnymi blokami SmallViT. |
| P5 | [Conservativeness of Untied Auto-Encoders — Im, Belghazi, Memisevic, AAAI 2016](https://ojs.aaai.org/index.php/AAAI/article/view/10268) | Zweryfikowany tekst źródłowy i DOI. Symetria Jacobianu i zachowawczość są znanym zagadnieniem. G16 oddziela lokalny test od istnienia globalnego potencjału. |
| P6 | [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods — Zhang, Nanda, wersja 2024](https://arxiv.org/html/2309.16042) | Odczyt metod, wyników i zaleceń: sposób korupcji oraz metryka mogą zmienić wnioski patchingu; Gaussian noise może wprowadzać OOD. Adaptacja do obrazu jest naszą propozycją, nie gotowym wynikiem dla ViT. |
| P7 | [How to use and interpret activation patching — Heimersheim, Nanda, 2024](https://arxiv.org/html/2404.15255v1) | Odczyt sekcji 2–4: kierunki noising/denoising nie są równoważne, działają na określonej rodzinie kontrastów; kontrolować metryki i zachowania kompensacyjne. Nie utożsamiać pojedynczego rescue z pełnym minimalnym obwodem. |
| P8 | [Sanity Checks for Saliency Maps — Adebayo i in., NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | Zweryfikowany rekord i abstrakt. Motywacja kontroli losowania modelu/danych; nie twierdzenie, że nasze pole jest saliency mapą. |
| P9 | [The emergence of clusters in self-attention dynamics — Geshkovski, Letrouit, Polyanskiy, Rigollet, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html) | Zweryfikowane metadane i abstrakt; istniejący nurt dynamiki tokenów. Nie przenosić asymptotycznej dynamiki uproszczonego attention na jednorazowy forward SmallViT. |
| P10 | [Transformer Block Coupling and its Correlation with Generalization in LLMs — Aubry, Meng, Sugolov, Papyan, ICLR 2025](https://openreview.net/pdf?id=kvLenbZZgg); [tekst arXiv v4](https://arxiv.org/html/2407.07810v4) | Odczyt abstraktu oraz definicji coupling, równanie 9; ponowna kontrola w v4 z 3 marca 2025. Ta wersja zmienia metrykę wobec v3. Artykuł obejmuje również ViT, więc porównanie tylko z językowymi pracami nie wystarczy. Wymagany baseline G28. |

Nie wykazano pierwszeństwa metody. Minimalny wkład ponad literaturę ma dotyczyć konkretnej funkcji pola i kontrolowanych interwencji, nie samego stosowania dynamiki/Jacobianów.

## C. Testy poprawności i geometrii

### G01 — tożsamość funkcji, modelu i zegara [obowiązkowy; PLAN E0]

- **Hipoteza techniczna:** miernik wykonuje tę samą deterministyczną funkcję co checkpoint, z jednoznacznym miejscem w grafie.
- **Procedura:** zapisać SHA checkpointu, kodu i danych; wersje torch/timm, klasy modułów, ε każdej LN, q/k norm, LayerScale, dropout/drop-path, dtype, TF32/fused attention, preprocessing, BN i tryb eval. Zarejestrować 13 stanów residualu i końcowy odczyt; osobno a,z MLP. Etykiety czasu to liczba aktualizacji i podkrok, nie sam indeks pliku.
- **Kontrole:** forward bez hooków, z hookami tylko do odczytu i ręcznie rozpisanym grafem; dwa identyczne powtórzenia; prawidłowe i błędne ustawienie eval jako kontrola instrumentu.
- **Jednostka:** X, wszystkie podkroki na małym audycie z E0.
- **Metryki:** maksymalny i względny błąd stanów/wyjść, zgodność dekodera, deterministyczność; manifest kompletności.
- **Falsyfikuje:** różnica większa od dopuszczalnego błędu numerycznego lub brak krytycznego metadatum. Wyników kolejnych G nie interpretować do naprawy instrumentu.
- **Zakres/koszt:** kilka F na punkt, bez treningu. Nie importować skryptów, które wykonują pomiar przy imporcie. P1/P2.

### G02 — podpisane wkłady MLP i biasy [obowiązkowy; E0]

- **Hipoteza:** Σz_cv_c+b₂ odtwarza rzeczywiste F; z768 nie jest stanem residualu x192.
- **Procedura:** porównać sumę wkładów z modułem MLP i różnicą stanów x⁺−x; uwzględnić rzeczywiste ls2/dropout. Sprawdzić dodatnie, ujemne i bliskie zeru z. Dla attention osobno sprawdzić sumę wkładów z v_k zawierającym b_V i b_O.
- **Kontrole:** pominięcie b₂/b_V/b_O, użycie |z| i pomylenie pre/post-LN powinny zostać wykryte. Permutacja neuronów wraz z dopasowaną permutacją obu warstw ma zachować wynik.
- **Jednostka:** T, również tło i tokeny o małej wypadkowej.
- **Metryki:** błędy F i x⁺; ||Σu||² kontra Σ||u||²+2Σ_{c<d}u_cᵀu_d; liczba ujemnych aktywacji jako opis.
- **Falsyfikuje:** niezgodność sumy poza tolerancją. To test algebry, nie teza, że wyrazy krzyżowe są energią fizyczną.
- **Zakres/koszt:** jeden F i O(md) na token; wszystkie 600 tokenów tanio w tym samym przebiegu.

### G03 — analityczna pochodna MLP i LN [obowiązkowy; E0]

- **Hipoteza:** wzór z A odtwarza DF konkretnego modułu.
- **Procedura:** obliczyć J_M analitycznie i przez autodiff na kopii w float64. Dla v sprawdzić Jv oraz dla w sprawdzić Jᵀw; potwierdzić iloczyn skalarny wᵀJv=vᵀJᵀw. Użyć rzeczywistej GELU/jej trybu approximation.
- **Kontrole:** prawdziwe ε kontra historyczne 1e−6; pełny DL kontra pominięty człon radialny i idealizowany rzut 190D; stany o małej oraz zwykłej wariancji kanałów.
- **Jednostka:** T, wybrane checkpointy i sześć MLP.
- **Metryki:** błędy J, JVP, VJP oraz adjoint identity; zmiana klasyfikacji kierunków przy przybliżeniu.
- **Falsyfikuje:** błędy pochodnej przekraczające tolerancję instrumentu. Wysokie R² rekonstrukcji aktywacji z historycznej odwrotności GELU nie zastępuje tej kontroli.
- **Zakres/koszt:** J_M+autodiff wybranych punktów; nie potrzeba pełnego J dla każdego tokenu. P1/P2.

### G04 — pełne pochodne attention przez Q, K i V [obowiązkowy; E0/E1]

- **Hipoteza:** pełny JVP attention jest sumą pochodnych trzech dróg, a obecny query-only jest tylko operatorem warunkowym.
- **Procedura:** zapisać attention jako h(Q(X),K(X),V(X)); policzyć osobno D_Qh·DQv, D_Kh·DKv, D_Vh·DVv przy pozostałych argumentach zamrożonych, następnie zsumować. Porównać z JVP całego oryginalnego modułu. Dla perturbacji tokenu j obserwować wszystkie wyjściowe tokeny i własny blok j→j.
- **Kontrole:** tryb explicit/fused, o ile wspierają pochodne; query-only z GUI; perturbacja własnego key/value; biasy. Wartość pola w centrum i jego pochodna to osobne kontrole.
- **Jednostka:** X; sondy pełnego stanu oraz oddzielnie sondy jednego tokenu.
- **Metryki:** błąd sumy dróg, udział każdej drogi w odpowiedzi z wyrazami krzyżowymi, mapa ||J_ijv|| i błąd query-only względem pełnej odpowiedzi.
- **Falsyfikuje:** brak zgodności pełnej sumy. Duża różnica query-only wyklucza wykorzystanie GUI-J jako substytutu pełnej dynamiki.
- **Zakres/koszt:** O(rD) na obraz, start r=8–16 zgodnie z benchmarkiem; bez macierzy 115200². P3/P4.

### G05 — dokładność numeryczna i centralne różnice [obowiązkowy; E0]

- **Hipoteza:** pochodne i efekty interwencji przekraczają poziom kwantyzacji oraz niedeterminizmu.
- **Procedura:** dla jednostkowych v liczyć d_h=[F(x+hv)−F(x−hv)]/(2h) i porównać z Jv na siatce h względem ustalonej skali tokenu/bloku. Oddzielić epsilon LN od kroku h. Szukać zakresu zbieżności O(h²), a następnie pogorszenia od zaokrągleń. Powtórzyć na kopiach float64/float32; historyczny float16 promowany do float32 traktować jako osobny warunek.
- **Kontrole:** affine F, dziedzina o małej normie Jv, brak/obecność TF32, dwa przebiegi identyczne, kilka losowych i kierunkowych v.
- **Jednostka:** T lub X, kierunek×h; nie niezależne próby biologiczne/statystyczne.
- **Metryki:** błąd absolutny i względny z jawnym dolnym progiem mianownika, krzywa błędu względem h, stosunek sygnał/błąd numeryczny.
- **Falsyfikuje:** brak zakresu stabilnej zgodności albo efekty porównywalne z błędem. Nie zmniejszać h bez końca i nie dostrajać tolerancji do pożądanego znaku.
- **Zakres/koszt:** 2rH F plus rD na podzbiorze, gdzie H liczba kroków h; planowane progi E0 pozostają wstępne.

### G06 — dokładny gauge i iloraz 191D [obowiązkowy; E0/E1]

- **Hipoteza:** przesunięcia c_j1 są neutralne dla każdego podkroku i niewidoczne dla końcowej głowy; radialna skala przy ε>0 nie jest dokładnie neutralna.
- **Procedura:** dodać niezależne c_j do kanałów każdego tokenu; sprawdzić F(X+g)=F(X), Ψ(X+g)−Ψ(X)=g, końcowe wyjścia identyczne. Sprawdzić Jg=0, Mg=g oraz formułę DLr. Zbudować U i porównać surowe wyniki z J̄=UᵀJU.
- **Kontrole:** wspólny i niezależny c_j, różne amplitudy ponad i poniżej naturalnej skali; idealizacja ε=0 tylko jako odrębny syntetyczny wariant; średnia po tokenach nie jest tym testem.
- **Jednostka:** X i T.
- **Metryki:** reszty tożsamości, widmo surowe/ilorazowe, odsetek efektu A przypadającego na sprzężenia z gauge.
- **Falsyfikuje:** naruszenie tożsamości przy poprawnym obecnym grafie wymaga diagnozy instrumentu/architektury. Nie wolno ogłaszać pełnej ścisłej pułapki przy zachowanym gauge.
- **Zakres/koszt:** kilka F/D; jeden U współdzielony. P1/P2; dokładny wniosek dla modelu wynika z algebry.

### G07 — zmiana bazy, metryka i permutacje [obowiązkowy; E0/E1]

- **Hipoteza:** deklarowane niezmienniki nie zależą od arbitralnej ortogonalnej bazy, a wyniki zależne od metryki są tak opisane.
- **Procedura:** dla stałej T badać funkcję F′(x′)=T F(T⁻¹x′), nie naiwny obrót wag przez GELU. Zweryfikować J′=TJT⁻¹. Dla ortogonalnej T porównać normy/widma S,A,M; dla anizotropowego T stosować G′=T⁻ᵀGT⁻¹ i adjoint J†=G⁻¹JᵀG. Osobno permutować ukryte neurony z obiema warstwami i biasem.
- **Kontrole:** nieprzetransformowana metryka jako celowo zależny wariant; ta sama baza dla wszystkich stanów kontra PCA dopasowywana osobno.
- **Jednostka:** T, transformacja współrzędnych; jedna metryka dopasowana na discovery.
- **Metryki:** błędy niezmienników, stabilność agregatów/rankingów; principal angles podprzestrzeni zamiast znaków eigenvectorów.
- **Falsyfikuje:** zmiana deklarowanego niezmiennika pod transformacją zachowującą jego definicję. Zmienność pod dowolnym skalowaniem nie dowodzi błędu S/A, lecz ogranicza claim.
- **Zakres/koszt:** algebra małych macierzy i kilka F; bez treningu.

### G08 — ruch odniesienia i centrowanie [obowiązkowy dla porównań z GUI]

- **Hipoteza:** przesunięcie środka chmury i skala wizualizacji są poprawnie oddzielone od lokalnego pola.
- **Procedura:** porównać surowe F_j, F_j−średnia_i F_i(X), oraz pole z zamrożoną średnią z X₀. Dla pojedynczej surowej perturbacji tokenu MLP δx_j, po ponownym centrowaniu δF_center,j=(1−1/N)J_jδx_j. Jednocześnie δx_center,j=(1−1/N)δx_j i pozostałe pozycje wycentrowane też się zmieniają: zawsze podać, względem której zmiennej liczy się pochodną. Dla attention użyć C_N DF pełnego stanu.
- **Kontrole:** stała średnia, ponownie liczona średnia, jedna stała skala s_ref kontra skala dopasowana po perturbacji. W GUI s_ref jest średnią norm, nie automatycznie RMS.
- **Jednostka:** X, ten sam punkt i interwencja.
- **Metryki:** błąd skoku, różnica Jacobianów wynikająca z referencji, rozdzielenie dryfu średniej i ruchu względnego.
- **Falsyfikuje:** przypisanie różnicy układów odniesienia zmianie mechanizmu lub utożsamienie zera wycentrowanego pola z F=0.
- **Zakres/koszt:** kilka F/D, wykorzystać G04 i G06.

### G09 — wiarygodność projekcji i brak domknięcia 3D [obowiązkowy; E1]

- **Hipoteza badana:** wskazane zjawisko w 3D odpowiada zdefiniowanej pełnowymiarowej własności, a nie tylko wybranemu rzutowi.
- **Procedura:** zamrozić P o ortonormalnych wierszach na niezależnym zbiorze; zestawić PJPᵀ z pełnym J lub J̄ oraz zmierzyć resztę (I−PᵀP)JPᵀ. Dla wielu bloków porównać PΦPᵀ z iloczynem PM_kPᵀ. Użyć wspólnego P oraz kilku z góry wybranych losowych projekcji. Przy nieortogonalnych osiach potrzebne są właściwe mapy podniesienia i metryka, nie samo Pᵀ.
- **Kontrole:** syntetyczny transport wychodzący poza rzut i wracający do niego; PCA każdej klatki tylko jako ilustracja artefaktu. Do pełnego attention użyć E/P w całym stanie, a nie ignorować innych tokenów.
- **Jednostka:** X lub T dla pojedynczego MLP, projekcja×kierunek.
- **Metryki:** utracona norma odpowiedzi, różnica kompozycji, zgodność znaków kierunkowego wzrostu; pełny versus rzutowany r-volume.
- **Falsyfikuje:** zanikanie odnośnego wniosku po przejściu do właściwego wymiaru. Brak zjawiska w innym rzucie nie dowodzi jego globalnego braku.
- **Zakres/koszt:** reuse J_M lub rD; nie tworzyć pełnego J attention.

### G10 — znane pola i kontrprzykłady [obowiązkowy; E0]

- **Hipoteza techniczna:** instrument rozróżnia gradient, obrót, ścinanie, wzrost przejściowy i właściwą stabilność dyskretną.
- **Procedura:** analityczne F=−kx; F=ωRx w 2D; F=(ky,0); F=Jx z J=[[-1,k],[0,-1]], k>2; mieszany blok z dokładnym neutralnym gauge. Sprawdzić wszystkie użyte wskaźniki.
- **Kontrole:** odwrócenie orientacji pętli, bias dodany do pola, obrót bazy, residual z krokiem h≠1 jako osobny model.
- **Jednostka:** przypadek syntetyczny, nie checkpoint.
- **Metryki:** znane S,A, loop integral, σ(M), eig(M), determinant i potęgi M. Dla −kI+ωR: stabilność I+J wymaga (1−k)²+ω²<1, mimo k>0.
- **Falsyfikuje:** instrument nazywający ścinanie orbitą zamkniętą albo nieokreślone S niestabilną iteracją w nilpotentnym przykładzie. To błędy interpretacji, nie odkrycia w sieci.
- **Zakres/koszt:** kilka małych macierzy CPU; bez ViT, planowane testy jednostkowe naukowego instrumentu.

### G11 — zakres lokalnej liniowości [obowiązkowy; E1/E2]

- **Hipoteza:** pierwszorzędowy model trafnie przewiduje odpowiedź w deklarowanym promieniu i klasie perturbacji.
- **Procedura:** dla δ=ηsv mierzyć R_F=F(x+δ)−F(x)−Jδ oraz R_Ψ. Siatka η z planu `[1e−4,3e−4,1e−3,3e−3,1e−2,3e−2]` jest startowa, nie gwarantowana. s=||Qx||; przy małym s zastosować zamrożony dolny próg skali z discovery. Kierunki: losowe, S̄/SVD, radialne, z perturbacji obrazów. Oba znaki.
- **Kontrole:** inicjalizacja, stan liniowy z G10, kwantyzacja z G05, kierunki wybrane bez końcowego błędu na TEST.
- **Jednostka:** T/X×kierunek×η, zagnieżdżone w P.
- **Metryki:** ||R||, relatywna reszta z bezpiecznym mianownikiem, nachylenie log–log, przedział promieni wiarygodności, reszta końcowego odczytu.
- **Falsyfikuje:** brak przewidywalności przy amplitudach użytych do tezy. Nie rozszerzać lokalnego mechanizmu na większe ruchy bez testu.
- **Zakres/koszt:** 2rH F plus pochodne; te same przebiegi służą G12/G20/G21.

### G12 — F kontra rzeczywista mapa I+J [obowiązkowy; E1]

- **Hipoteza:** wniosek o wzmocnieniu/kontrakcji pozostaje poprawny dla skończonego residualu.
- **Procedura:** liczyć K=MᵀM−I=2S+JᵀJ, singular values M, kierunkowe g(v)=||Mv||/||v|| i porównanie z prawdziwą nieliniową parą. Powtórzyć w ilorazie. Dla lokalnego MLP obliczyć sign/logabsdet, bez exponentiating skrajnych logów.
- **Kontrole:** klasyfikacja po samym S, trace J i rho(M); kontrprzykłady G10. Przy niezerowym F(x₀) nie mówić o stabilności punktu x₀.
- **Jednostka:** T, kierunek i η z G11.
- **Metryki:** odsetek kierunków z ujemnym vᵀSv, lecz dodatnim vᵀKv; σ_max/min, log|det M|, znak determinanty, rzeczywiste g_η.
- **Falsyfikuje:** tezę „ujemne S oznacza przywracanie w jednym kroku”, jeśli różni się od zmierzonego residualu. rho<1 nie gwarantuje monotonicznego spadku każdej normy.
- **Zakres/koszt:** SVD/eig małych J_M; pełne attention tylko operatorowo. P4 i własna algebra.

### G13 — pełny transport przez pozostałą sieć [obowiązkowy; E1/E2]

- **Hipoteza:** kompozycja pochodnych odtwarza całą odpowiedź, w tym wpływy przez inne tokeny.
- **Procedura:** propagować E przez każdy M_k; porównać ΦE z bezpośrednim JVP złożenia i skończonymi parami obrazów/latentów. Ocenić C_E, singular values ΦE, końcowy BΦE oraz udział odpowiedzi poza początkowym tokenem. Zachować kolejność attention→MLP i miejsce końcowej LN.
- **Kontrole:** iloczyn w odwróconej kolejności, zerowanie cross-token bloków, iloczyn rzutów 3D, sam iloczyn norm poszczególnych M jako bound, nie równość.
- **Jednostka:** X×punkt startowy×r kierunków.
- **Metryki:** błąd chain rule, rzeczywiste wzmocnienie, r-volume, przepływ czułości do sąsiednich/dalekich tokenów.
- **Falsyfikuje:** odpowiedź niezgodna z pełną kompozycją. Jeśli lokalna poprawa po bloku ginie później, nie wolno nazywać jej końcową korzyścią detektora.
- **Zakres/koszt:** rD przez suffix, O(Ndr) na sondy; r startowo 16, test zbieżności 8/16/32 na podpróbie. Bez treningu.

### G14 — czułość głowy, poziomice i zadaniowy odczyt [obowiązkowy dla claimów o bbox]

- **Hipoteza:** wskaźnik geometrii wnosi informację o końcowym odczycie, nie tylko o ruchu w niewidocznych kierunkach.
- **Procedura:** dla końcowego stanu uzyskać B=DH (5×192), z SVD i rzeczywistym epsilon. Dla wcześniejszego stanu użyć pochodnej całego suffixu. Mierzyć EᵀΦᵀBᵀRBΦE, przy R zamrożonym na discovery. Porównać perturbacje w ker B i jego ortogonalnym dopełnieniu; dla większych ruchów testować krzywiznę/ponowną projekcję na H=stała.
- **Kontrole:** gauge1, losowe kierunki o tej samej normie, głowa przyłożona do niewłaściwego miejsca tylko jako błędna kontrola. Przy niepełnym rzędzie nie wpisywać automatycznie 187.
- **Jednostka:** T/P, wyjście stałego tokenu; ranking dekodera osobno.
- **Metryki:** singular values B, reszta Bδ, reszta nieliniowa, błąd pięciu wyjść, rzeczywisty błąd GT. Przy rank5 lokalna poziomica ma187 wymiarów, po ilorazie186.
- **Falsyfikuje:** deklarowaną funkcję zadaniową, jeśli duże geometryczne efekty pozostają niewidoczne dla odczytu albo nie poprawiają zdefiniowanego endpointu. BᵀRB jest półokreślone i nie może być odwracane jak pełna metryka.
- **Zakres/koszt:** pięć VJP dla wybranych pięciu wyjść lub rJVP; nie budować globalnego Gramianu. Istniejąca dolina to ilustracja lokalna, nie dowód osiągalności obrazu.

### G15 — wkład neuronów do przywracania i cyrkulacji [E1/E3]

- **Hipoteza:** wybrane grupy neuronów odpowiadają za zadaniowo istotną orientację lokalnej pochodnej, ponad sumę ich aktualnych wektorów.
- **Procedura:** dla ustalonych jednostkowych kierunków e,f w residualu liczyć dokładnie κ_c(e)=−ϕ′(a_c)(eᵀv_c)(w_cᵀDL e) i Γ_c(e,f)=ϕ′(a_c)[(fᵀv_c)(w_cᵀDL e)−(eᵀv_c)(w_cᵀDL f)]. Sprawdzić Σκ_c=−eᵀSe oraz ΣΓ_c=fᵀJe−eᵀJf=2fᵀAe. Dla analizy ilorazowej podstawić e=Uē,f=Uf̄. Ranking/grupę wybrać na discovery; ocenić osobno jej wkład do pola, pochodnej i odczytu. Późniejszy patch G24 ma sprawdzić znaczenie funkcjonalne.
- **Kontrole:** losowe grupy tej samej liczebności i siły, ranking po |z|, ||u_c|| i czułości głowy; permutacja neuronów z G07. Zachować znaki GELU i ϕ′; nie zastępować sum wkładów sumą ich modułów.
- **Jednostka:** T×kierunek/płaszczyzna; nie 768 niezależnych prób.
- **Metryki:** błąd sum, koncentracja i kompensacja κ/Γ, stabilność grup, przewidywanie skończonego efektu odczytu poza baseline'ami.
- **Falsyfikuje:** brak rekonstrukcji, brak przewagi grupy na holdout albo brak przewidzianego efektu interwencji. Duża κ nie dowodzi stabilności I+J.
- **Zakres/koszt:** jeden J_M lub jego składniki i O(md) na kierunek; współdzielić G02/G03/G24, bez treningu.

### G16 — cyrkulacja w jednym zamrożonym polu [E1; konieczny dla terminu „niezachowawczość”]

- **Hipoteza:** lokalna antysymetria przewiduje rzeczywistą cyrkulację dla małych pętli w ustalonej metryce.
- **Procedura:** ustalić checkpoint, podkrok, kontekst pozostałych tokenów, bazę i skalę. Całkować F·dx po prostokątach lub kołach w płaszczyźnie ortonormalnych e,f, z dodatnią orientacją e→f. Dla malejącego rozmiaru sprawdzić ∮F·dx/area → fᵀJe−eᵀJf. Porównać kwadratury o rosnącej rozdzielczości. W metryce stałej G właściwą formą jest (GF)ᵀdx, z pochodną GF. Dla attention poza badaną płaszczyzną zamrozić cały pozostały kontekst i jawnie nazwać wynik przekrojem pełnego pola.
- **Kontrole:** odwrócona orientacja, stały bias, pole gradientowe i czysty obrót G10; wspólna baza. Nie łączyć odcinków z różnych checkpointów ani podkroków w jedną „pracę”.
- **Jednostka:** T/X×stała płaszczyzna×promień.
- **Metryki:** reszta względem 2 area fᵀAe, błąd kwadratury, stabilność po przeskalowaniu z poprawną metryką.
- **Falsyfikuje:** brak zbieżności do przewidzianej cyrkulacji. Niezerowy wynik wyklucza lokalny potencjał euklidesowego pola w otoczeniu pętli; zerowy wynik na skończonej liczbie pętli nie dowodzi globalnego potencjału.
- **Zakres/koszt:** liczba węzłów kwadratury×F, tylko wybrane płaszczyzny z G15. Jednostki są jednostkami ustalonego latentu, bez kalibracji nie dżulami. To nie test orbit zamkniętych ani prądów prawdopodobieństwa. P5.

### G17 — równowaga i stabilność, tylko jeśli pozostaje claim o pułapce [opcjonalny; E1]

- **Hipoteza:** w określonej dziedzinie istnieje rzeczywisty punkt stały badanego podkroku lub jego mapy ilorazowej.
- **Procedura:** rozdzielić trzy zadania: F(x*)=0; UᵀF(Uq*)=0; P F(x*)=0. Drugie daje fixed point w ilorazie i może zachować dryf gauge, trzecie tylko zero przekroju. Poszukiwać w zadanym trust region od naturalnych stanów i walidować resztę oryginalną funkcją. Dla znalezionego punktu badać I+J, a w ilorazie I+J̄, następnie krótkie kontrolowane iteracje tej SAMEJ mapy jako sztucznego modelu pomocniczego.
- **Kontrole:** pole wycentrowane i surowe, zmiana startu i promienia, bias, kontrprzykłady G10. Nie nazywać ujemnego/nieokreślonego S siodłem równowagi bez punktu stałego i właściwego widma.
- **Jednostka:** kandydat×stały blok×checkpoint; niezależność startów nie zastępuje niezależności obrazów.
- **Metryki:** ||F||, ||UᵀF||, odległość od naturalnych stanów, residual solvera, widmo mapy i skończone odpowiedzi. Kryterium rho<1 dotyczy lokalnej asymptotycznej stabilności znalezionego punktu dyskretnej mapy; rho>1 daje kierunek niestabilny; przypadek brzegowy wymaga dalszej analizy.
- **Falsyfikuje:** duża pełna reszta albo pozorny korzeń tylko po projekcji. Nieznalezienie korzenia przez optimizer nie dowodzi nieistnienia; znalezienie poza wsparciem danych nie dowodzi używania go przez sieć.
- **Zakres/koszt:** opcjonalny ograniczony solver z F/D; przerwać po zadanym budżecie. Nie jest warunkiem publikacji o funkcjonalnej geometrii. Pełna ścisła pułapka jest wykluczona przez gauge; neutralność nie wyklucza stabilności poprzecznej.

### G18 — realistyczne perturbacje obrazów i właściwy GT [obowiązkowy; E2]

- **Hipoteza:** lokalna geometria przewiduje odpowiedź na zdefiniowaną rodzinę zmian wejścia istotną dla zadania.
- **Procedura:** wspólne pary obrazów z głównego protokołu: nuisance o niezmienionym GT oraz perturbacje semantyczne/geometriczne z prawidłowo przekształconym GT. Rejestrować rzeczywiste ΔX na wszystkich tokenach i w kolejnych miejscach. Porównać przewidywanie suffixu Df·ΔX z rzeczywistą odpowiedzią, uwzględniając resztę G11. Interwencje G20–G24 stosować do tych samych par. Transformacje GT i maski ważności sprawdzić niezależnie od predykcji.
- **Kontrole:** identyczny obraz, odwrócona/zerowa transformacja, perturbacje o tej samej normie w latencie, zakres amplitud poniżej i powyżej lokalnej liniowości, clipping/krawędzie obrazu. Translacja nie oznacza automatycznie niezmienności bbox ani indeksu zwycięskiego tokenu.
- **Jednostka:** P, klaster filmu i seeda według katalogu głównego.
- **Metryki:** główny błąd GT targeted−sham, błąd przewidywania końcowej odpowiedzi, fixed-token versus rzeczywisty decoder, czułość do nuisance i semantyki oddzielnie.
- **Falsyfikuje:** brak przewidywania poza discovery albo przewaga tłumienia wszystkich zmian, w tym potrzebnych dla poprawnego przesunięcia ramki. Sama mniejsza norma ΔX nie jest lepszą detekcją.
- **Zakres/koszt:** wspólne pary i forwardy katalogu głównego, bez oddzielnego iloczynu wszystkich siatek G. P3/P6.

### G19 — perturbacje ambientowe i kontrola OOD [obowiązkowy przy latent surgery]

- **Hipoteza:** efekt syntetycznej perturbacji jest interpretowalny w zbadanym promieniu i nie wynika wyłącznie z opuszczenia typowych stanów.
- **Procedura:** porównać naturalne ΔX z G18, kierunki losowe, kierunki widmowe i pojedynczy token. Skalę dopasować w jednej ustalonej metryce, raportować liczbę perturbowanych tokenów oraz profil norm. Dopasować na discovery diagnostyki odległości do stanów: lokalną PCA/subspace residual, nearest-neighbor distance i ewentualnie regularyzowaną Mahalanobis. Raportować także wariancję kanałów, normę, aktywacje i confidence. Nie dopasowywać rozkładu referencyjnego na TEST.
- **Kontrole:** szum poza empiryczną podprzestrzenią przy tej samej normie, łagodna perturbacja obrazu, zerowa interwencja; te same progi odległości dla targeted i sham. Gaussian noise jest osobną rodziną, nie automatycznie realistyczną korupcją.
- **Jednostka:** P×rodzina; tokenowe sondy są analizą pomocniczą.
- **Metryki:** endpoint, reszta Taylora, odległości referencyjne, udział wyjść poza zadeklarowaną dziedzinę oraz czułość wyniku na filtr OOD.
- **Falsyfikuje:** claim o naturalnym mechanizmie, jeśli efekt występuje tylko dla odległych syntetycznych stanów. Mała odległość do skończonej próby nie dowodzi leżenia na rzeczywistej rozmaitości obrazów ani istnienia obrazu realizującego dany latent.
- **Zakres/koszt:** reuse G11/G18 i tanie diagnostyki; nie rozwiązywać inwersji obrazu dla każdego punktu. P6/P7.

### G20 — lokalna interwencja w antysymetrię z zachowaniem gauge [główny test przyczynowy; E2]

- **Hipoteza:** orientacja Ā w określonym punkcie zmienia odpowiedź zadaniową względem odpowiednio dobranej kontroli. Kierunek korzyści ustalić na development i zamrozić przed TEST.
- **Procedura:** dla naturalnego x₀ zamrozić Ā₀ i zdefiniować F_α(x)=F(x)−αχ(||Uᵀ(x−x₀)||/r) U Ā₀ Uᵀ(x−x₀), gdzie χ jest gładkie, r zadeklarowane, a χ=1 blisko zera. Wtedy F_α(x₀)=F(x₀) i J̄_α(x₀)=S̄₀+(1−α)Ā₀. Użyć α=0,0,5,1 lub węższej siatki zatwierdzonej po pilocie; nie zmieniać centrum przy każdym perturbowanym wejściu. Zmierzyć odpowiedź całego suffixu. Pierwszy rząd różnicy końcowej wynosi −α DR(Ψ(x₀))U Ā₀Uᵀδ. Zakres reszty sprawdza G11.
- **Kontrole:** normmatched sham G22, efekt numeryczny α=0, niezmienność wyniku bazowego, zachowanie gauge, kontrola S/G21 i dawka. Surowe usunięcie A może złamać gauge: J1=0 nie implikuje A1=0. Użyć ilorazu, a dla attention jego pełnostanowego odpowiednika U_N.
- **Jednostka:** P×zamrożone miejsce interwencji; wybór miejsc tylko development.
- **Metryki:** główny paired endpoint targeted−sham, zgodność z liniową prognozą, dawka–odpowiedź, rzeczywista norma poprawki i OOD G19, efekty fixed-token/decoder.
- **Falsyfikuje:** brak przewidzianego efektu ponad sham, odwrócenie prerejestrowanego znaku na niezależnym materiale lub efekt istniejący tylko poza wiarygodnym promieniem. Zmiana endpointu po dużej dowolnej deformacji nie potwierdza specyficznej roli A.
- **Zakres/koszt:** MLP korzysta z J_M. Dla attention działanie Ā bez macierzy: Q_N(J₀−J₀ᵀ)Q_Nδ/2, czyli jeden JVP i jeden VJP w zamrożonym X₀; wszystkie 600 tokenów pozostają w operatorze. To lokalna sztuczna modyfikacja funkcji, nie globalne „usunięcie wiru” z wag ani dowód istnienia orbit. P4/P6/P7.

### G21 — interwencja w deformację S i rzeczywiste wzmocnienie mapy [E2]

- **Hipoteza:** wybrane kierunki deformacji mają określoną funkcję zadaniową odrębną od samej normy poprawki oraz A.
- **Procedura:** z S̄=VΛVᵀ zbudować S̄_+=V max(Λ,0)Vᵀ i S̄_−=V min(Λ,0)Vᵀ. Zastąpić Ā₀ w konstrukcji G20 wybraną składową S̄_+/S̄_− lub ustaloną podprzestrzenią. Dodatkowo dla twierdzenia o wzroście w jednym kroku wybrać kierunki z K̄=M̄ᵀM̄−I. Jawnie odróżnić modyfikację operatora S̄ od sondowania kierunków K̄. Nie zakładać, że usunięcie dodatniego S̄ musi zmniejszyć każdą singular value M̄.
- **Kontrole:** S-sham G22, A-interwencja przy podobnej dawce, losowe kierunki, stała wartość F(x₀), analogiczne promienie i rodziny perturbacji; kontrola rzeczywistego J po interwencji.
- **Jednostka:** P×podprzestrzeń wybrana przed TEST.
- **Metryki:** paired endpoint, zmiana g(v), σ(M̄), orientacji odpowiedzi względem B, reszta nieliniowa i dawka. Efekty S_+ i S_− raportować osobno, bez wybierania po wyniku lepszego wariantu.
- **Falsyfikuje:** błędną prognozę wzrostu mapy lub brak zadaniowej przewagi nad sham. Nieokreślone S nadal nie dowodzi siodła punktu stałego.
- **Zakres/koszt:** reuse J_M i G20; pełne attention tylko wybrane kierunki/operatorowe przybliżenia z G27. Bez dodatkowych treningów.

### G22 — czy sham kontroluje rzeczywistą alternatywę? [obowiązkowy dla G20/G21/G24]

- **Hipoteza techniczna:** efekt targeted jest porównywany z ingerencją o zadeklarowanej sile, a nie z łatwiejszą kontrolą.
- **Procedura:** w ilorazie losować ortogonalne O niezależnie od TEST i stosować R_A=O Ā₀Oᵀ (analogicznie dla S). Zachowuje to klasę skew/symmetric, Frobenius norm, rank i widmo danego składnika; nie zachowuje automatycznie normy całego J ani singular values I+J. Mierzyć rzeczywistą ||Rδ|| i ||Āδ||. Dla głównego normmatched porównania dopasować dawkę w przestrzeni poprawki przed suffixem według zamrożonej reguły. Sham per δ, przeskalowany przez ||Āδ||/||R_Aδ||, jest kontrolą skończonej dawki; na ogół nie jest jednym liniowym operatorem o tej samej pochodnej w centrum. Tych interpretacji nie mieszać.
- **Kontrole:** wiele ustalonych losowań, self/sham identity, dolny próg mianownika i jawne liczby zdegenerowanych przypadków; oddzielnie raportować kontrolę operatorową i kontrolę dawki. Nie dopasowywać do końcowego błędu GT, który jest wynikiem interwencji.
- **Jednostka:** P×targeted/sham, losowania zagnieżdżone, nie niezależne przykłady.
- **Metryki:** równowaga norm, liczebności, miejsc i dawek, OOD, rozkład paired endpoint; czułość do liczby sham. Dobór kontroli zamrozić po pilocie, nie po końcowym znaku efektu.
- **Falsyfikuje:** wniosek o orientacji, gdy przewaga znika po dopasowaniu rzeczywistej siły lub wynika z degeneracji mianownika albo nierównego budżetu perturbacji.
- **Zakres/koszt:** małe losowe macierze dla MLP i współdzielone forwardy. Dla pełnego stanu stosować strukturalne izometrie/operatorowe shamy bez gęstego O; strukturalne ograniczenia losowania muszą być opisane.

### G23 — pary tokenów i kierunkowe sprzężenie attention [E2/E3]

- **Hipoteza:** wybrane połączenie między tokenami wnosi kierunkowy wpływ do lokalizacji ponad siłę i położenie perturbacji.
- **Procedura:** dla pełnego attention J_ij=∂F_i/∂x_j porównać eᵀJ_ij f i fᵀJ_ji e. Blok antysymetrii pełnego J to A_ij=(J_ij−J_jiᵀ)/2, nie skew samego J_ij. Obejmuje wszystkie drogi Q/K/V. Wybrać pary development, porównać sondy źródła oraz skutki w odbiorcy i reszcie tokenów. Dla interwencji wycinać tylko zadeklarowany liniowy blok w konstrukcji zachowującej centrum i gauge; kolejne attention w suffixie pozostawić rzeczywiste.
- **Kontrole:** pary o podobnej odległości przestrzennej, normie i sile drogi, kierunek odwrotny, self-token, query-only, losowe pary, pełny versus wybrany blok. Asymetria J nie jest naruszeniem prawa Newtona: w kodzie nie ma takiego prawa oddziaływania.
- **Jednostka:** P×wybrana para lub mały zestaw; jednostka statystyczna nadal film/seed.
- **Metryki:** kierunkowa czułość, lokalna asymetria, udział skutku przechodzący przez inne tokeny, paired endpoint oraz BΦδ. Same attention weights nie są tymi miarami.
- **Falsyfikuje:** brak efektu poza dopasowaną kontrolą lub przypisanie kierunkowego wpływu Q, gdy pochodzi z K/V czy późniejszych bloków.
- **Zakres/koszt:** JVP/VJP ze wsparciem sond w wybranych tokenach; pełny blok 192×192 tylko dla małej liczby par, nie 600² bloków. Bez interpretacji hydrodynamicznej/fizycznej siły par. P3/P4.

### G24 — patching grup wskazanych przez geometrię [E3; wspólny z katalogiem interwencji]

- **Hipoteza:** grupy wybrane przez κ/Γ lub zadaniową orientację niosą funkcję określoną na danej rodzinie clean/corrupt, ponad prostsze selektory.
- **Procedura:** cache'ować jednoznacznie a,z lub stan residualu. W corrupt podmienić wybrane współrzędne na clean, a w odrębnym eksperymencie clean na corrupt; wykonać rzeczywisty suffix. Dla MLP patch z768 nie jest bezpośrednim patchem x192. Przy translacji rozstrzygnąć korespondencję tokenów według transformacji przestrzennej. Użyć tych samych grup i par co G15/G18; selekcja bez użycia efektu TEST.
- **Kontrole:** self-patch, losowe grupy o tej samej liczebności i normie zmiany, ranking po aktywacji/czułości, niepowiązany donor, dopasowana pozycja tokenu; zero/mean ablation jako osobne potencjalnie OOD interwencje. Denoising i noising mierzyć oddzielnie, nie wybierać korzystniejszego po wyniku.
- **Jednostka:** P×grupa×kierunek patcha.
- **Metryki:** główny endpoint, bezwzględne zmiany logitów i bbox, recovery fraction tylko przy stabilnym mianowniku, margines zwycięzcy i decoder. Ta głowa używa sigmoid, więc nie przenosić do niej niezmienniczości softmax na wspólne przesunięcie logitów.
- **Falsyfikuje:** brak przewagi nad selektorem bazowym lub rescue występujące jedynie dla OOD/dużej ingerencji. Rescue pokazuje warunkowy udział w danym kontraście; nie dowodzi uniwersalnej konieczności, wystarczalności ani minimalnego obwodu. Brak efektu może wynikać z kompensacji, którą trzeba sprawdzić osobno.
- **Zakres/koszt:** współdzielony główny katalog patchingu, nie oddzielna kampania; jedno cache pary plus forward na wybrane grupy/kontrasty. P6/P7.

### G25 — zegar uczenia: zmiana pola a zmiana stanu [obowiązkowy przy claimach „podczas uczenia”]

- **Hipoteza:** obserwowany trend geometrii daje się rozdzielić na zmianę parametrów i odwiedzanych stanów, zamiast utożsamiać go z ruchem w głębokości.
- **Procedura:** dla sąsiednich zapisów i tego samego obrazu/podkroku obliczyć D_ab=F_{θ_a}(X_b), a,b∈{0,1}. Δtotal=D_11−D_00 jest sumą Δweights=D_10−D_00, Δstate=D_01−D_00 oraz interaction=D_11−D_10−D_01+D_00. Powtórzyć dla J, używając zgodnej bazy. Alternatywny symetryczny podział po połowie interakcji zdefiniować z góry, nie mieszać go z powyższym. Rzeczywisty ruch reprezentacji po aktualizacji przewidywać przez D_θf·Δθ+D_bf·Δb i resztę, jeśli dostępne są właściwe stany i bufory BN.
- **Kontrole:** stałe X oraz stałe θ oddzielnie, dropout eval, ten sam preprocessing, faktyczna liczba kroków pomiędzy zapisami; stan optimizer/BN oraz przyszłe batch/RNG wspólne w ewentualnym osobnym teście restartu. Głębokość i checkpoint lag są osobnymi osiami.
- **Jednostka:** obraz×para checkpointów×seed, z autokorelacją w czasie.
- **Metryki:** norma i kierunek trzech składników, reszta rachunkowa, zmiana geometrii na wspólnym banku stanów, przewidywanie rzeczywistego Δx.
- **Falsyfikuje:** claim o zmianie wyuczonego operatora, gdy trend wynika z samego przesunięcia stanów; claim o dynamice tokenu x_(t+1)=x_t+F_M(x_t), gdy rzeczywista zmiana pochodzi z całej aktualizacji parametrów.
- **Zakres/koszt:** cztery oceny na wybraną parę oraz reuse pochodnych; bez nowych treningów w tym etapie. Przyczynowy restart optimizer to osobny koszt katalogu głównego. Token sam nie jest automatycznie stanem Markowa.

### G26 — trained versus init i niezależna replikacja [obowiązkowy; E4]

- **Hipoteza:** zadaniowy związek geometrii jest efektem uczenia i uogólnia się poza punkty użyte do jego odkrycia.
- **Procedura:** zestawić inicjalizację i checkpointy na własnych naturalnych stanach oraz w osobnym układzie na wspólnych stanach z G25. Zachować rozdział: trained weights na init states i odwrotnie mogą być OOD. Porównywać cały algorytm selekcji grup/kierunków pomiędzy seedami, nie identyczny numer neuronu. Zamrożone główne kontrasty ocenić na trzech seedach potwierdzających i niezależnym materiale; s47 oraz dwa rozwojowe pozostają discovery/development według głównego protokołu.
- **Kontrole:** losowe/losowane warstwy, normy i czułość, coupling G28, pozycja tokenu i obiekt/tło. Ewentualny trening na losowych etykietach stanowi osobny wariant wymagający budżetu; nie zakładać jego wykonania. Dopasowanie końcowej jakości może usuwać część badanego efektu i wymaga osobnej interpretacji.
- **Jednostka:** seed i film; checkpoint/token/neuron to pomiary zagnieżdżone.
- **Metryki:** efekty paired endpoint i ich heterogeniczność, przyrost przewidywania poza baseline'ami, kontrasty trained−init, odsetek replik zgodnych z wcześniej ustalonym znakiem.
- **Falsyfikuje:** brak różnicy od inicjalizacji lub zanik na niezależnym materiale. Geometria czy kompensacja istniejące równie silnie w init pozostają własnością konstrukcji/selektora, dopóki nie wykazano ich wyuczonej funkcji.
- **Zakres/koszt:** wykorzystuje autorytatywną kampanię seedów katalogu głównego. Nie powtarzać historycznych biegów tylko dla ścieżek. P8 motywuje sanity checks; nie stanowi dowodu skuteczności konkretnej kontroli.

### G27 — operatorowa estymacja widma i wiarygodność wyników [obowiązkowy dla pełnego attention]

- **Hipoteza techniczna:** oszacowania pełnostanowe mają wystarczającą dokładność dla zgłoszonej różnicy i nie udają certyfikowanych maksimów.
- **Procedura:** dla niezależnych sond o E[zzᵀ]=I estymować ||J||_F² przez średnią ||Jz||² oraz ||S||_F²,||A||_F² przez sondy (Jz±Jᵀz)/2. W ilorazie stosować sondy z=U_Nξ i projekcję wyniku, E[zzᵀ]=Q_N, co estymuje właściwy operator ilorazowy. Dla skrajnych singular/eigenvalues użyć iteracji operatorowych z restartami i residualami. Na małych J_M porównać z pełnym SVD/eig. Skonfrontować r=8/16/32 na wybranym podzbiorze; większe r tylko jeśli błąd uzasadnia koszt.
- **Kontrole:** syntetyczne widma G10, adjoint identity G03/G04, różne sondy/restarty/dtype, bliskie degeneracje. Porównywać principal angles podprzestrzeni zamiast dowolnych znaków i kolejności prawie równych eigenvectorów.
- **Jednostka:** operator×punkt×losowanie sondy; losowania estymatora nie są biologiczną/statystyczną replikacją modelu.
- **Metryki:** błąd względem dokładnego małego operatora, wariancja sond, residual iteracji, stabilność wniosków przy r/dtype. Trace estimate nie zastępuje log|det(I+J)|.
- **Falsyfikuje:** różnica porównywalna z błędem estymacji lub niestabilna wobec liczby sond. Wynik power iteration jest estymacją, zwykle dolnym przybliżeniem maksimum, nie dowodem globalnego sufitu; nonnormalne J może mieć wrażliwe eigenvalues.
- **Zakres/koszt:** O(rD) i O(Ndr) pamięci; bez pełnego J. Oddzielnie raportować błąd estymatora, błąd pochodnej i niepewność między filmami/seedami.

### G28 — bezpośredni baseline transformer block coupling [obowiązkowy dla nowości]

- **Hipoteza:** proponowana orientacja zadaniowa lub przyczynowa interwencja wnosi informację ponad znaną zgodność bloków, top singular values i prostą czułość.
- **Procedura:** odtworzyć wersję v4/P10 dla aktualizacji całego bloku attention+MLP bez zewnętrznego skipu: f_block(X)=F_A(X)+F_M(X+F_A(X)). Zatem J_block=(I+J_M)(I+J_A)−I=J_A+J_M+J_MJ_A, z J_M w stanie po attention. Baseline obejmuje sześć całych bloków, nie tylko 12 podkroków. Przy J₁=U₁Σ₁V₁ᵀ,J₂=U₂Σ₂V₂ᵀ: A_K=U₂,KᵀJ₁V₂,K, m_K=||A_K−Σ₁,K||_F/||s₁,K||_p i c_K=1−m_K, równanie 9; ustalić p/K oraz regułę zerowego mianownika. Wersja v3 używa innej miary i nie jest zamiennikiem. Warianty pojedynczych podkroków, ilorazu oraz całego operatora oznaczyć jako rozszerzenia. Oryginalne wybrane bloki token→token, własny blok attention oraz pełny operator nie są równoważne; jawnie podać, który porównujemy. Dla pełnego operatora top-K uzyskać przez JVP/VJP.
- **Kontrole:** σ_max, ||J||_F, normy aktualizacji, czułość BΦ, confidence/margines, warstwa i etap uczenia; pary bloków bliskie/dalekie, losowe dopasowania podprzestrzeni. Dopasowanie modeli predykcyjnych i wybór K wyłącznie development; wynik porównać na holdout.
- **Jednostka:** para bloków×obraz×checkpoint; agregacja i potwierdzenie po filmie/seedzie.
- **Metryki:** wiernie odtworzony coupling, przyrost przewidywania głównego endpointu, zgodność kierunkowych prognoz i efekt interwencji po dopasowaniu baseline'ów. c_K może być ujemne; nie jest prawdopodobieństwem. Nierozróżnialne singular subspaces wymagają kontroli zależności od wyboru bazy SVD.
- **Falsyfikuje:** claim o nowym użytecznym mechanizmie, jeśli wynik opisuje wyłącznie znane coupling/czułość i nie daje niezależnego, replikowalnego kontrastu przyczynowego. Sama analiza J lub ViT nie stanowi różnicy wobec P10.
- **Zakres/koszt:** reuse J_M i sond G27, tylko wybrane pary. Nie tworzyć dodatkowej pełnej kampanii. Odczyt dokładnej definicji i jej test syntetyczny poprzedzają implementację; P10 obejmuje ViT pomimo tytułu o LLM.

## D. Minimalna kolejność i granice wnioskowania

1. **Brama instrumentu:** G01–G10 na małym zbiorze z checkpointami/warstwami obejmującymi przypadki trudne. Przy niezgodności funkcji lub pochodnej wstrzymać interpretację dalszych wyników, nie uznawać ich za dowody ujemne mechanizmu.
2. **Jedna wspólna baza pomiarów:** G11–G16, G27/G28 korzystają z tych samych stanów, operatorów i sond. S/A, wartości singularne, cyrkulacja i coupling są różnymi pytaniami do tej samej funkcji, nie osobnymi treningami.
3. **Rdzeń przyczynowy:** G18–G22 na wspólnych parach z katalogu głównego, z G13/G14 jako transportem do zadania. G23/G24 tylko dla faktycznych claimów o parach/grupach. G17 jest opcjonalny, dopóki nie twierdzimy, że znaleziono pułapkę lub siodło równowagi.
4. **Oddzielenie uczenia:** G25 przy twierdzeniu o organizacji podczas treningu; cross-depth correction z katalogu głównego nie jest checkpoint lag. G26 potwierdza transfer między materiałem i seedami. Wszystkie duże wybory, znaki, dawki, miejsca i reguły sham zamrozić po pilocie przed TEST.

Nie wykonywać pełnego iloczynu wszystkich warstw×checkpointów×kierunków×promieni×sham. Najpierw brama poprawności, potem wybór stabilnego zakresu na development, na końcu mały zestaw zadeklarowanych kontrastów na confirmation. Podpróby obliczeń służące drogiej geometrii muszą być wybierane bez końcowego efektu TEST i posiadać jawną regułę agregacji.

**Najważniejsze luki przed wykonaniem:** aktualny przekrój attention z GUI nie jest pełnym J; starsze pomiary mają różne ε/precyzję/centrowanie; brak gotowego dowodu mapowania realistycznych perturbacji na zadaniowo istotne kierunki; dobór grup może sam produkować kompensację także w init. Istnieje bliska literatura coupling/Jacobianów także dla ViT. Dlatego publikację powinien nieść replikowalny wynik interwencji przewidujący konkretną zmianę detekcji, z poprawnym GT i kontrolą siły, a nie sama nazwa „wir”, „siodło” czy „dolina”.

**Granica tego dokumentu:** protokół został opracowany z odczytu kodu, istniejących raportów, literatury i algebry. Nie wykonano G01–G28 w tej sesji; nie uruchomiono modelu, treningu ani nowych pomiarów, i nie zmieniono aplikacji. Ten raport nie przypisuje proponowanym testom statusu ukończonych wyników.
