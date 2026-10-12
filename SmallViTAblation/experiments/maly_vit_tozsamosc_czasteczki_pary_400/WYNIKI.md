# Reprezentacja narzędzia: cząsteczki i makrowektory w 400 parach

Ukończono: 2026-10-01T21:47:44.659212+02:00. Analiza wszystkich 400 oryginałów i 400 odpowiadających edycji LaMa na jednym zamrożonym checkpointcie. **Mierzymy odpowiedź reprezentacji na usunięcie narzędzia, nie ruch podczas treningu.**

Wybrana na walidacji podprzestrzeń ośmiu kierunków daje na odłożonych filmach AUROC **96.86 procent** i zbilansowaną trafność **88.61 procent**. Osobny klasyfikator LDA osiąga 95.28 procent. Interwencja w osiem kierunków obniża confidence×IoU bardziej niż losowe kierunki, również po dopasowaniu normy po fc2. Przedział zmiany AP50 obejmuje jednak zero. Wynik wskazuje użyteczny sygnał obecności i wpływ na głowę detektora; nie dowodzi wyłącznej tożsamości cystotome.

## Dane, aparat i protokół

400 par z 116 filmów. Podział przed dopasowaniem: 69 filmów / 240 par train, 23 / 75 validation, 24 / 85 test, seed 20261001. Oba człony pary oraz wszystkie klatki filmu pozostają razem. Te obrazy należą do train detektora: test oznacza odłożenie do identyfikacji podprzestrzeni, nie niezależny heldout detektora. Filmy nie są potwierdzonymi identyfikatorami pacjentów.

Model uczony od zera, seed 47; k1000.pt to 1000 aktualizacji źródłowej rozgrzewki, nie numer epoki ani ogólnie pretrained ViT. Pełne klatki resize INTER_AREA do 480×320, RGB /255 i wewnętrzna normalizacja ImageNet. Eval, stałe BN, float32, brak autocast i TF32. Warstwa pomiaru: GELU ostatniej MLP, 600 tokenów po 768 współrzędnych.

Pomiar 800 obrazów trwał 42.71 s. Zachowano lossless float32 wszystkie **6729 × 2 × 768** wektory tokenów oryginalnej maski oraz 3200 × 2 × 768 kontrolnych tokenów odległego tła. TOKENY_PARY.npz zajmuje 70.07 MB. W obu członach stosujemy dokładnie te same indeksy maski ORYGINAŁU. Nie zastąpiono cząsteczek ich średnią; średnie ważone służą dodatkowo klasyfikacji i opisowi osi. Pełne tło wszystkich 600 tokenów można odtworzyć skryptem, bez zapisu wielkiego tensora.

Powtórny forward dał maksymalny błąd 0; wszystkie BN buffers pozostały identyczne. Nie było NaN ani pustych masek. Kod modelu jest identyczny z wcześniej zweryfikowanym snapshotem pomiaru pełnych tokenów. RTX 3070, PyTorch 2.6.0+cu124, timm 0.9.16; bez treningu, Eden jobs i commitów.

Model SHA256: `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`. Kod modelu: `0570c957a23a75283102806160fa197c5ba66c27776d05b27a3ed461929da57a`. Manifest danych: `47358bad3fb1768e07020d2d3791a5f27a66e5259bcb859ea2efe80d87c69687`. Zapis tokenów: `07193aa2ede29953040ec9634cdf8acb596c23624e6223fe46cafb9e6d2b7088`. Pozostałe hashe, wersje i czasy w POMIAR.json / WYNIKI.json; pochodzenie LaMa i masek w źródłowym eksperymencie.

## Które makrowektory reagują

Cząsteczka to jeden token z_i w wspólnej przestrzeni R^768. Makrowektory e_j są wspólnymi osiami bazowymi, a mikrowektory to składowe z_ij e_j. Kierunek związany z konceptem może być kombinacją wielu osi. Poniżej ranking **powtarzalnej odpowiedzi**, ustalony na train; nie indywidualnie udowodniona przyczynowość osi.

Delta = oryginał minus LaMa. Średnie tokenów są ważone pokryciem maski; statystyki obrazów mają równe wagi filmów, a w filmie równe wagi obrazów. Znak minus oznacza większą składową po edycji. Wartości mają arbitralne jednostki aktywacji sieci.

| Oś (1-based) | Indeks od 0 | Średnia delta train | SD | Średnia/SD | Zgodność znaku filmów train | Zgodność znaku obrazów test |
|---:|---:|---:|---:|---:|---:|---:|
|500|499|+0.15752|0.10363|+1.520|100.0%|96.5%|
|472|471|-0.25907|0.17118|-1.513|100.0%|96.5%|
|735|734|-0.27388|0.18315|-1.495|100.0%|97.6%|
|512|511|-0.26639|0.18234|-1.461|100.0%|96.5%|
|647|646|-0.16358|0.12257|-1.335|98.6%|92.9%|
|326|325|+0.10042|0.07780|+1.291|100.0%|96.5%|
|461|460|+0.10766|0.08467|+1.272|100.0%|95.3%|
|166|165|+0.07397|0.05854|+1.264|95.7%|92.9%|
|553|552|+0.07221|0.05717|+1.263|98.6%|95.3%|
|445|444|-0.13854|0.11157|-1.242|97.1%|94.1%|

![Najbardziej spójne reagujące osie](MAKROWEKTORY_TOP20.png)

Kandydatów SVD wyznaczono z niescentrowanego drugiego momentu różnic tokenów train, zachowując średni kierunek. Wagi: film, obraz, pokrycie. r50 = 5, r90 = 88; osiem wybranych kierunków obejmuje 60.53 procent energii delty. To energia zmian, nie procent tożsamości semantycznej.

![Energia delty](ENERGIA_PODPRZESTRZENI.png)

Pełna baza U jest w PODPRZESTRZEN.npz. Największe wagi każdego mieszanego kierunku:

| Kierunek | Osie e_j: podpisana waga (indeks od 0) |
|---:|---|
|1|e226: +0.172 (225); e655: -0.161 (654); e80: -0.160 (79); e390: -0.155 (389); e6: -0.147 (5)|
|2|e383: -0.147 (382); e496: -0.145 (495); e42: -0.142 (41); e729: -0.130 (728); e85: -0.123 (84)|
|3|e655: -0.150 (654); e305: +0.146 (304); e717: -0.126 (716); e735: -0.118 (734); e182: -0.117 (181)|
|4|e160: -0.145 (159); e305: +0.137 (304); e483: +0.133 (482); e369: -0.125 (368); e223: +0.119 (222)|
|5|e23: +0.222 (22); e217: +0.183 (216); e660: +0.176 (659); e122: +0.163 (121); e758: +0.147 (757)|
|6|e23: +0.221 (22); e438: +0.175 (437); e305: +0.173 (304); e243: -0.146 (242); e674: -0.130 (673)|
|7|e226: -0.237 (225); e23: -0.198 (22); e100: -0.165 (99); e496: +0.158 (495); e438: -0.150 (437)|
|8|e59: -0.197 (58); e323: -0.148 (322); e472: -0.105 (471); e89: -0.104 (88); e350: -0.102 (349)|

Największe wagi osobnego kierunku LDA: e655: -0.1460 (od 0: 654); e647: -0.1338 (od 0: 646); e577: -0.1248 (od 0: 576); e221: -0.1171 (od 0: 220); e472: -0.1151 (od 0: 471); e512: -0.1135 (od 0: 511); e176: +0.1101 (od 0: 175); e445: -0.1058 (od 0: 444); e576: -0.0999 (od 0: 575); e500: +0.0995 (od 0: 499). Kierunek klasyfikacji nie jest automatycznie kierunkiem zapisu informacji.

## Obecność poza dopasowaniem

Wybór SVD: rank 1/2/4/8/16 i logistic C 0.01/0.1/1/10; maksimum film-weighted val AUROC, bez testu. Wybrano rank 8, C = 1. Próg 0.585208 dobrano wyłącznie na validation. Standardyzacja probe używa zwykłej średniej/SD obrazów train; baza i strata logistic mają wagi filmowe. Nie każdy etap transformacji jest film-weighted.

| Metoda | Val AUROC | Test AUROC | Test zbilansowana trafność |
|---|---:|---:|---:|
|SVD 8 + logistic|97.49%|96.86%|88.61%|
|LDA ridge 1|98.67%|97.94%|95.28%|

95-procentowe przedziały film-bootstrap (500 powtórzeń): SVD AUROC [94.18, 99.10] procent, trafność [83.37, 93.30] procent. LDA trafność [90.94, 98.68] procent. Image-weighted SVD trafność 88.82 procent.

Etykieta to oryginał versus syntetyczne usunięcie; ROI maski oryginału jest znany w obu obrazach. Nie testujemy tu rozpoznawania innego narzędzia ani niezależnego identyfikatora egzemplarza.

KATY.json zawiera 12 binów natywnej osi anotacji, po 2–10 par testowych. Trafność per bin 75.0–100.0 procent. Leave-angle-out refituje U, standardyzację i probe bez danego binu; rank/C odziedziczone z podstawowego val wszystkich kątów. To diagnostyka częściowego transferu, nie rygorystyczny unseen-angle benchmark. 19/400 obrazów zmienia bin po affine resize; kąt wejścia zapisano osobno.

## Wpływ na detekcję i dwie rodziny kontroli

Wszystkie ramiona oceniają te same 85 oryginałów i GT. Bbox manifestu XYWH jawnie przekształcono do XYXY. Interwencja na GELU: z_o − UUᵀ(z_o − z_n) tylko w tokenach oryginalnej maski. Pełna podmiana z_n stanowi drugie ramię. Trzy losowe ortonormalne podprzestrzenie rank 8, seeds 11/22/33, dopasowano najpierw do normy perturbacji PER TOKEN w 768D, następnie w osobnych ramionach do normy PER TOKEN po fc2. U pozostaje zamrożone.

Po badanej GELU nie ma kolejnej attention: fc2, residual, LayerNorm i głowa są punktowe. Forward faktycznie dokończono do detektora. Decode max-pool i top-k uruchomiono ponownie. Noop jest dokładnie identyczny. GT center leży w tool ROI dla 80/85 obrazów, globalny peak dla 60/85. Pozostałe tokeny mogą nadal zawierać informację zgromadzoną przez wcześniejszą attention.

| Ramię | AP50 | Strata | confidence×IoU | Norma perturbacji 768D | Norma po fc2 |
|---|---:|---:|---:|---:|---:|
|nohook|45.76%|2.48568|0.36232|0.0000|0.0000|
|shamnoop|45.76%|2.48568|0.36232|0.0000|0.0000|
|identity|44.11%|2.47685|0.34303|15.4947|8.5665|
|full_patch|41.70%|2.48657|0.34068|20.8335|10.0207|
|random11|45.07%|2.48725|0.36097|15.4947|4.5236|
|random22|46.07%|2.48155|0.36124|15.4947|4.4995|
|random33|45.45%|2.48421|0.35941|15.4947|4.7206|
|fc2random11|41.77%|2.49066|0.35607|29.4318|8.5665|
|fc2random22|45.23%|2.47830|0.35855|29.6212|8.5665|
|fc2random33|41.94%|2.48463|0.35525|28.1641|8.5665|

Identity obniża confidence×IoU o -0.01929, 95% paired film CI [-0.02763, -0.01230]. Kontrast identity minus średnia trzech random dopasowanych w 768D: -0.01751, CI [-0.02444, -0.01065].

Po dopasowaniu normy po fc2 przewaga w tej mierze pozostaje: identity minus średnia trzech fc2-random **-0.01359**, CI **[-0.01922, -0.00842]**. Random wymaga wtedy większej normy w 768D (28.16–29.62 versus 15.49). Maksymalny błąd dopasowania normy po fc2: 4.77e-07; zero degeneracji. Wykonano 255 dodatkowych rzeczywistych forwardów, 23.20 s. Dopasowanie normy nie dopasowuje orientacji perturbacji względem residual/LN/głowy; wynik nadal nie dowodzi wyłącznej semantyki.

**Brak rozstrzygnięcia AP50 i straty:** identity ΔAP50 -1.65 punktu procentowego, 95% CI [-6.24563, +4.11311] punktu. Δstrata -0.00883, CI [-0.03995, +0.02229]. Oba przedziały zawierają zero. Nowe fc2-random dają AP50 41.77/45.23/41.94 procent; nie ma konsekwentnej przewagi identity w AP.

AP50 to niezmieniona source funkcja jednej klasy, interpolacja 101-punktowa, pooled po obrazach; bootstrap klastrowy po filmach, 200 powtórzeń. Strata i confidence×IoU mają równe wagi filmów. Wszystkie ramiona używają tego samego GT. Paired hook korzysta z obserwowanej wersji LaMa; jest diagnostyczną interwencją, nie predyktorem bez znanego negatywu.

## Które cząsteczki na konkretnym obrazie

CZASTECZKI_RANKING.json zachowuje ranking reakcji dla każdego obrazu: pełna delta L2, jej projekcja, pokrycie i indeks tokenu. CZASTECZKI_WPLYW.json mierzy osobno zmianę metryki po interwencji pojedynczej cząsteczki w sześciu przykładach wybranych po kątach przed oglądaniem ich wyników. Podmiana jednej surowej komórki output z mapy all-identity jest tu równoważna pojedynczemu hookowi; dwie rzeczywiste kontrole dały maksymalny błąd 0.

**Przykład source 8, TR_0053_S1_P03/00000881:** największa projekcja zmiany jest w cząsteczce **280 (indeks od 0: 279)**, około 6.984. Jej indywidualna zmiana confidence×IoU wynosi 0. Największy spadek tej metryki występuje w cząsteczce **248 (od 0: 247)**: −0.02845. Duża reakcja latentna nie musi oznaczać dużego wpływu na ostatnią głowę. Oba tokeny oznaczono na planszy HEATMAP_008.png.

| Source | Największa projekcja: cząsteczka (od 0), row/col | Największy wpływ: cząsteczka (od 0), row/col | Δconfidence×IoU |
|---:|---|---|---:|
|8|280 (279), 9/9|248 (247), 8/7|-0.02845|
|80|377 (376), 12/16|254 (253), 8/13|+0.00000|
|143|500 (499), 16/19|500 (499), 16/19|-0.04162|
|201|555 (554), 18/14|315 (314), 10/14|+0.00000|
|288|254 (253), 8/13|282 (281), 9/11|-0.01216|
|353|407 (406), 13/16|379 (378), 12/18|-0.01634|

![Reakcja i indywidualny wpływ: source 8](HEATMAP_008.png)

![Reakcja i indywidualny wpływ: source 80](HEATMAP_080.png)

![Reakcja i indywidualny wpływ: source 143](HEATMAP_143.png)

![Reakcja i indywidualny wpływ: source 201](HEATMAP_201.png)

![Reakcja i indywidualny wpływ: source 288](HEATMAP_288.png)

![Reakcja i indywidualny wpływ: source 353](HEATMAP_353.png)

## Tło, sham i ograniczenia semantyczne

Na teście średnia projekcja tool ROI = 3.77760, odległe 8 tokenów tła = 1.66231; pełna delta L2 odpowiednio 5.07969 i 2.28873. Zmiana może rozchodzić się przez attention na całą scenę; maska nominalnego patcha nie zapewnia semantycznej czystości późnego tokenu.

Główna kontrola **24 sham wyłącznie z testowych filmów**, dwa przykłady w każdym binie natywnego kąta: ta sama maska edycji przesunięta na tło, narzędzie pozostaje. Tool ROI po sham: delta L2 0.25051, projekcja 0.15922. Po usunięciu narzędzia na tych samych źródłach: 4.84873 i 3.56190. Sham edit ROI: 0.83674 i 0.50176.

Sham zmienia średnio probability(original) − probability(sham) o -0.01646; największe bezwzględne przesunięcie 0.27341. Nie zmieniła się binarna decyzja w żadnej z 24 par, lecz klasyfikator **reaguje**. W 13/24 edycja i maska narzędzia dzielą nominalny token 16×16 mimo rozłączności pikseli; odpowiedź tool ROI częściowo obejmuje tę wspólną komórkę.

Pierwsze 24 mixed sham (18 train / 3 val / 3 test) raportujemy osobno, bez łączenia powtarzających się source_index z główną kontrolą. Podmienione tło nie jest prawdziwym tłem spod narzędzia. Sham zmienia inną treść i pozycję; inne instrumenty nie mają tutaj kontrolowanych masek. Kontrola osłabia proste wyjaśnienie samą edycją, ale nie usuwa wszystkich artefaktów LaMa ani nie dowodzi cystotome versus inne narzędzia.

## Wniosek i zakres

Zidentyfikowano powtarzalne reagujące osie i podprzestrzeń użyteczną do paired presence oraz wykazano jej większy wpływ na confidence×IoU niż losowe kierunki przy obu kontrolach normy. Zachowano negatywny wynik dla jednoznacznej zmiany AP50/straty. LDA lepiej rozdziela pary niż SVD 8, lecz wpływu kierunku LDA na detektor nie testowano. Nie wykazano wyłącznej tożsamości semantycznej, identyfikacji egzemplarza, całej tożsamości w 768D ani przyszłego ruchu podczas uczenia. Potrzebne inne narzędzia, niezależne obrazy i dodatkowe rodzaje edycji.

Przed końcową oceną Astra wykryła konwencję bbox XYWH; poprawiono do XYXY. Pierwszy bieg hook przerwał się na indeksach int16 bez zapisanych metryk; poprawiono torch.long i cały test policzono od nowa. Wszystkie publikowane tu metryki detekcji pochodzą z poprawnego biegu. Podprzestrzeń nie zależała od bbox. Różne role informacji i ograniczenia zapisano przed analizą w PLAN.md.

## Artefakty i odtwarzanie

Pomiar: POMIAR.json, TOKENY_PARY.npz, PODZIAL.json. Baza/probe: PODPRZESTRZEN.npz, IDENTYFIKACJA.json, MAKROWEKTORY.json. Reakcja i wpływ tokenów: CZASTECZKI_RANKING.json, CZASTECZKI_WPLYW.json, sześć HEATMAP. Detekcja: DETEKCJA.json, DETEKCJA_FC2.json, ich per-image JSON i surowe mapy. Sham: osobne SHAM_DIAGNOSTYKA_mixed/heldout.json i lossless TOKENY_SHAM. Audyt: AUDYT_ASTRA.md. Kod i hashe: WYNIKI.json.

Uruchomić kolejno z F:/Python/python.exe w tym katalogu: pomiar.py, identyfikacja.py, interwencje.py, kontrola_fc2.py, raport_koncowy.py. Nie uruchamiać raport_koncowy przed kompletnością kontroli. Źródłowych zdjęć ani wag nie zmieniono. PODSUMOWANIE.md należy do roota i nie jest nadpisywane.
