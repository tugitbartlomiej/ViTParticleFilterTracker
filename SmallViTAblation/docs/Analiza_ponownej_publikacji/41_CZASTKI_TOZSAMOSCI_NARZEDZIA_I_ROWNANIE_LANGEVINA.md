# 41. Które cząstki niosą tożsamość narzędzia i co opisuje w nich rozszerzone równanie Langevina

Data: 2026-09-22. Status: **uzasadnienie teoretyczne i protokół diagnostyczny; bez nowych pomiarów.**
Odpowiada na pytanie autora: „trzeba zdiagnozować, które cząstki niosą tożsamość narzędzia, i sprawdzić, jak je
opisuje moje rozszerzone równanie Langevina; co mi to daje w obrazie i w śledzeniu narzędzia".
Równolegle, tego samego dnia, powstał audyt [40](40_TOZSAMOSC_NARZEDZIA_I_ROLA_ROZSZERZONEGO_LANGEVINA.md)
(inna sesja): rozdziela obecność, typ, ciągłość celu i egzemplarz, wypisuje człony równania grantu tak, jak stoją
w kodzie, i projektuje tracker z modelem obserwacji. Ten dokument nie powtarza tamtych treści; dokłada definicję
„cząstek niosących tożsamość" przez domknięcie dynamiki (twierdzenia Lean z grantu i Mori–Zwanzig), sortowanie
cząstek po skali czasu i protokół pomiaru po tokenach. Przy rozbieżności liczb o raporcie K4 wiąże audyt 40
(10 współrzędnych rzeczywistych, kierunek pomiaru „geometria → współrzędne").
Rozwija dokumenty [36](36_KOOPMAN_LANGEVIN_TOZSAMOSC_NARZEDZIA_I_DOBOR_KLATEK.md) (równanie na latencie i torze),
[38](38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md) (stan narzędzia w latencie detektora, filtr)
i [39](39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md) (wolne mody całej klatki to odcisk operacji, nie narzędzia).
Równanie w wersji źródłowej: projekt grantowy, `experiments/tft_langevin_krok_po_kroku/README.md`,
`tft_langevin_czlon_zdarzen`, `tft_langevin_optymalizacja_95`; twierdzenia w Lean:
`Research/Math/lean4_spm/SpmProofs/{MoriZwanzig,DiagnostykaInnowacji}.lean`; postulaty o cząstkach:
`experiments/czastki_i_pole/TEORIA.md` i `Research/DiagnosticTool/teoria/`.

## 0. Czego ten dokument NIE twierdzi

Nie twierdzi, że równanie Langevina poprawi detektor ani tracker; etap 1 doboru klatek (dokument 39) pokazał, że
jako selektor klatek sygnał Langevina nie pomaga. Nie twierdzi, że w zbiorze trackingowym da się zmierzyć tożsamość
MIĘDZY narzędziami: jest tam jedno narzędzie (cystotom), więc „tożsamość" znaczy tu „to, co w cząstkach narzędzia
zostaje stałe, gdy zmienia się położenie". Nie podaje nowych liczb; wszystkie liczby pochodzą z dokumentów 36, 38, 39
i z eksperymentów grantu.

## 1. Słownik: trzy znaczenia słowa „cząstka" i jedno obowiązujące tutaj

| Gdzie | Cząstka to | Położenie cząstki to | Krok czasu to |
|---|---|---|---|
| grant, `tft_langevin_*` | jedna para sklep–produkt | latent TFT okna 56 dni (192 liczby) | jeden dzień |
| teoria grantu, T6 | jeden token (patch) obrazu po zamianie na wektor | wektor tokenu | jedno przejście przez warstwę (T33: klatka to NADKROK) |
| dokument 38 | jedna klatka | pooling latentu detektora w ramce | jedna klatka (zegar filmu) albo jeden checkpoint (zegar treningu) |

Tu obowiązuje znaczenie z teorii: **cząstka = token**, bo pytanie „które cząstki niosą tożsamość narzędzia" ma sens
tylko wtedy, gdy cząstek w jednej klatce jest wiele i różnią się rolą. Tokenów jest 256 na klatkę w cache DINOv2
(`feature_cache/koopman_tokens`, 8 klipów, 410 klatek × 256 tokenów × 48 wymiarów po projekcji, krok 5 klatek)
i `fh × fw` na mapie cech RF-DETR (dziś zapisywany jest tylko ich pooling: `box_pool`, `global_pool`, 256 wymiarów,
4800 klatek). Postulat T32 mówi, że tożsamość cząstki niesie ADRES (miejsce na siatce), nie treść. Narzędzie porusza
się PO adresach, więc „cząstka narzędzia" nie może być jednym adresem; musi być zbiorem tokenów sklejonym treścią.
To jest pierwsza rzecz, którą równanie ma rozstrzygnąć: **które adresy w danej klatce są członkami tej samej,
zamkniętej trajektorii**. Postulat T6a dodaje higienę: nie każdy token jest cząstką; rejestry (tokeny o anomalnie
wysokiej normie, które przestają reprezentować swój adres) trzeba wykluczyć zanim policzy się cokolwiek po tokenach.

## 2. Twierdzenie, które porządkuje całość: pamięć jest ceną rzutowania

Rozszerzone równanie z grantu, bez wymuszenia zewnętrznego (film nie ma wejścia znanego z góry):

```text
z_{t+1} = z_t + D(z_t, v_t, a_t) + JP(z_{t-1}, z_{t-2}, ...) + N(dawne reszty) + e_{t+1}
```

`D` to dryf z bieżącego stanu, prędkości i przyspieszenia; `JP` to jądro pamięci po wcześniejszych krokach; `N` to
przewidywalna część dawnych reszt; `e` to reszta (innowacja). Formalizm Moriego–Zwanziga mówi, skąd biorą się `JP` i `e`:
gdy z dużego układu obserwuje się tylko część zmiennych, ich dokładna dynamika ma trzy człony: markowowski (nasze `D`),
całkę pamięci po własnej przeszłości (nasze `JP`) i człon ortogonalny zależny od nieobserwowanych zmiennych (nasze `e`).
Lin, Tian, Anghel, Livescu 2021 (10.1137/21m1401759), równanie 22, zdanie na s. 11: „pamięć i siła zewnętrzna istnieją
tylko dlatego, że zbiór obserwabli jest niekompletny"; oba człony znikają, gdy obserwable są zamknięte pod dynamiką.
Dla układu liniowego jądro ma jawną postać `K(s) = A_12 exp(s A_22) A_21` (Wang, Benner, Heiland 2026, arXiv:2606.23341,
równanie 11): iloczyn sprzężenia „obserwowane → ukryte", propagacji wśród ukrytych i sprzężenia „ukryte → obserwowane".
Ten sam fakt w wersji dowiedzionej w grancie: `projekcja_rodzi_pamiec` (obserwacja jednej współrzędnej ukrytego układu
2D spełnia rekurencję z jądrem `-det A`), `jadro_projekcji_znika_iff` (jądro znika wtedy i tylko wtedy, gdy ukryty układ
jest efektywnie jednowymiarowy), `mz_zero_iff_warunkowa` (przeszłość nic nie dodaje wtedy i tylko wtedy, gdy warunkowa
kowariancja przyszłości z przeszłością przy danym stanie jest zero).

Wniosek definicyjny, który zamienia pytanie autora w pomiar:

> **Zbiór cząstek niesie tożsamość narzędzia wtedy, gdy jest zamknięty pod własną dynamiką**: równanie dopasowane do
> samych tych cząstek ma jądro pamięci `JP` nieodróżnialne od zera ORAZ resztę `e` białą (nieskorelowaną z niczym
> z historii), a dołożenie cząstek sceny ani nie zmniejsza `JP`, ani nie wybiela reszty.

Oba warunki są potrzebne. `kryterium_bialosci` (Lean): reszta jest prostopadła do całej historii wtedy i tylko wtedy,
gdy model jest optymalny; korelacja reszty z czymkolwiek z historii dowodzi brakującego członu. `kontrprzyklad_D59`
(Lean): istnieje proces, w którym jawne jądro pamięci wychodzi DOKŁADNIE zero, a pamięć siedzi w kolorze reszty
(autokowariancja na lagu 2 równa `r^4 > 0`). Zanik jądra bez białej reszty nie dowodzi domknięcia. Dokładnie to
zdarzyło się w grancie: odczepy `N` wybieliły resztę (autokorelacja z −0,16 do +0,09), a wynik łączny nie drgnął;
w filmie: innowacje filtru mają korelację 0,17 do 0,20 na lagu 1 i kurtozę 25 do 132 (dokument 38, 4c), więc stan
`[p, v, a]` z samej ramki NIE jest zamknięty. Czegoś brakuje i równanie mówi, że to coś jest w cząstkach, których
filtr nie widzi.

Drugie twierdzenie, `rozklad_bledu`: błąd dowolnego predyktora z historii to zaskoczenie (nieusuwalne, prostopadłe do
historii) plus część usuwalna. To pozwala rozstrzygnąć, czy słaba przewidywalność ruchu narzędzia (22 do 29 procent
na torach GT, dokument 36) to sufit fizyczny, czy brak cząstek w stanie. Odpowiedź daje ten sam test domknięcia: jeśli
po dołożeniu cząstek reszta bieleje, a skill rośnie, brakowało cząstek; jeśli reszta jest już biała, reszta to ręka
chirurga i żaden model z historii jej nie odzyska.

## 3. Co to daje w OBRAZIE (jedna klatka, wiele cząstek)

### 3.1. Sortowanie cząstek według skali czasu: tożsamość, stan, ruch

Operator Koopmana dopasowany do trajektorii cząstek porządkuje kierunki latentu według tempa zmian. Klus i in. 2017
(10.1007/s00332-017-9437-7): współrzędne TICA to funkcje własne operatora Koopmana rzutowane na bazę liniową
(sekcja 3.2), a wartości własne najbliższe 1 to najwolniejsze procesy (sekcja 2.4, `1 = λ_1 > λ_2 ≥ ...`). Analiza
wolnych cech (Wiskott, Sejnowski 2002, 10.1162/089976602317318938) porządkuje cechy „według stopnia niezmienniczości"
i Franzius, Wilbert, Wiskott 2011 (10.1162/NECO_a_00171) pokazują, że taki porządek rozdziela tożsamość obiektu,
położenie i kąt obrotu. To daje obraz, o który pyta autor:

| Skala czasu | Wartość własna | Co to jest dla narzędzia | Gdzie to już zmierzono |
|---|---|---|---|
| niezmiennik | `\|λ\| = 1` | tożsamość: które to narzędzie, jego wygląd | nie zmierzono (jedno narzędzie w zbiorze) |
| wolne | `\|λ\| > 0,98` | stan: położenie, rozmiar | tor GT: 4 mody wolne, 72 procent ładunku na położeniu (dokument 36, 3.2); latent `box_pool`: 6 wolnych modów niesie położenie z R² 0,73 do 0,87 wobec 0,04 do 0,45 dla PCA (dokument 38, 4a) |
| szybko gasnące | `\|λ\| << 1` | prędkość, drganie | tor GT: mody gasnące, 74 procent ładunku na prędkości |
| poza równaniem | brak | wymuszenie: ręka chirurga | innowacje w rzadkich skokach, kurtoza 25 do 132 (dokument 38, 4c) |

Równanie nie dodaje tu nic ponad Koopmana z jednym wyjątkiem: mówi, że jeśli w cząstkach narzędzia zostanie pamięć
`JP`, to hierarchia jest niepełna i brakuje wolnej zmiennej, którą trzeba dopisać do stanu (`markowizacja_ar2`, Lean:
pamięć skończonego rzędu daje się wchłonąć do stanu przez rozszerzenie). To jest konstrukcyjna droga, którą grant
znalazł w literaturze jako najbliższy precedens: She i in. 2022 (10.1063/5.0130033) uczą „cechy niemarkowowskie"
razem z rozszerzoną dynamiką; Lei, Baker, Li 2016 (10.1073/pnas.1609587113) zamieniają jądro na bieguny, które
definiują zmienne pomocnicze; Gupta i in. 2023 (10.48550/arXiv.2310.10745) doczepiają pamięć do latentu autoenkodera
i pokazują, że przy 16 obserwablach zysk z pamięci znika (sekcja 4.1.1). Ten ostatni wynik jest ważnym falsyfikatorem:
jeśli `box_pool` w 24 wymiarach ma już `JP ≈ 0`, analiza po tokenach nic nie doda.

### 3.2. Ostrzeżenie z własnych pomiarów: wolne mody CAŁEJ klatki to scena, nie narzędzie

Dokument 39, sekcja 5d, po korekcie z audytu 40 (`REPORT_koopman_cover.json`: 8 wybranych funkcji własnych daje
10 współrzędnych rzeczywistych, nie 16): wolne współrzędne Koopmana latentu DINOv2 całej klatki mają 72 procent
wariancji MIĘDZY filmami i korelację klatka do klatki 0,94, a regresja z geometrii narzędzia na te współrzędne
(kierunek „geometria → współrzędne", liniowa) wyjaśnia około zera. To odcisk palca operacji: oko, światło,
ustawienie mikroskopu. Nie wyklucza to nieliniowej informacji o narzędziu w tych współrzędnych; wyklucza, że są
one jego stanem w sensie sekcji 2. W języku sekcji 2: cząstki sceny są zamknięte i mają własną tożsamość, a narzędzie jest dla nich
szumem. Dlatego pytanie „które cząstki" jest pytaniem o PODZIAŁ populacji tokenów, nie o wybór kierunków w jednym
uśrednionym wektorze. Dokument 36, 3.1, pokazał też, że sam podział „to, co się rusza, to narzędzie" (DMD bez wolnych
modów) trafia w narzędzie 0,353 wobec 0,398 dla zwykłej różnicy klatek: rusza się także oko, drugie narzędzie i odblaski.
Kryterium domknięcia z sekcji 2 jest inne: nie „co się rusza", tylko „co rusza się jako jedno zamknięte ciało, którego
następny krok przewiduje jeden wspólny dryf z białą resztą". To jest hipoteza do sprawdzenia, a wynik DMD jest
ostrzeżeniem, że może wyjść źle.

### 3.3. Higiena: rejestry

Darcet i in. 2023 (10.48550/arXiv.2309.16588): w DINOv2 tokeny o normie powyżej 150 (2,37 procent tokenów ViT-g)
tracą informację o własnym położeniu (trafność 22,8 wobec 41,7 procent) i o pikselach, a niosą informację globalną;
pojawiają się tylko w modelach ViT-L i większych, na brzegach mapy, w jednolitym tle. Lappe, Giese 2025
(10.52202/085713-0037): w dużych ViT token klasy czyta głównie z rejestrów, a nie z tokenów obrazu. Trzon RF-DETR Nano
ma rozmiar ViT-S, więc rejestrów najpewniej nie ma, ale test T6a (norma przed normalizacją końcową plus stałość
indeksu) jest tani i musi poprzedzać każdą statystykę po tokenach; tokeny wykluczone to nie cząstki.

## 4. Co to daje w ŚLEDZENIU (ciąg klatek)

### 4.1. Prędkość istnieje tylko w ciągu; równanie jest modelem procesu dla filtru

RF-DETR widzi jedną klatkę. Prędkości w jednym latencie nie ma (dokument 38, sekcja 1; sonda prędkości z jednej klatki
R² 0,00; Becker i in. 2019, arXiv:1905.07357: „niemożliwe uzyskać prędkość z jednego obrazu"). Prędkość i prawo ruchu
żyją w ciągu cząstek, a równanie Langevina jest tym prawem zapisanym w postaci, którą filtr może użyć:

```text
p' = p + v
v' = f_v v + a
a' = f_a a + w        w ~ ciężkoogonowe wymuszenie (ręka)
y  = p + n            n = błąd odczytu położenia
```

To jest `kalman_state.py`. Wynik (dokument 38, 4c, wagi 2500): różnica skończona położenia daje R² −0,53, filtr
przyczynowy +0,44, wygładzacz +0,22; z latentu `box_pool` odpowiednio −1,88, +0,04, −0,52, bo liniowy odczyt położenia
z latentu ma błąd 11 pikseli, a ramka 3. Wersja gaussowska filtru zawiodła (korelacja 0,10 do 0,31), wersja odporna
(Huber, obcięcie innowacji) działa. Co z tego ma tracker:

1. **prędkość, której detektor nie ma**, i to prędkość z jakością rosnącą z uczeniem detektora (0,49 → 0,67);
2. **skalibrowane „nie wiem"**: rozmiar innowacji jako niepewność do ustawienia szerokości skanu iOCT między klatkami;
3. **sufit predykcji**: z historii toru da się przewidzieć 22 do 29 procent następnego kroku (dokument 36, 3.2), pamięć
   `JP` dodaje 1 do 2 punktów, `N` nic, bo reszta toru jest już biała. Około trzech czwartych ruchu to zaskoczenie,
   czyli ręka. To jest liczba, której żaden tracker z historii nie przeskoczy; horyzont predykcji między klatkami
   trzeba projektować pod nią, nie pod nadzieję.

Uwaga metodyczna z literatury trackingowej: dobrze dostrojony filtr o stałej prędkości potrafi zniwelować pozorną
przewagę wyuczonych modeli ruchu (Greenberg i in. 2021, semanticscholar 680b6327); nasz pomiar pokazuje odwrotność
(stała prędkość −169 do −181 procent na torach GT, bo kroki ramki są przeciwskorelowane), więc odniesieniem musi być
filtr z bezwładnością i odpornym szumem, nie stała prędkość.

### 4.2. Tożsamość przez zasłonięcie: to, co ma przeżyć, gdy narzędzia nie widać

Przy zasłonięciu (końcówka pod tęczówką, drugie narzędzie, odblask) detektor milczy, a tracker musi trzymać dwie rzeczy
osobno: STAN (gdzie narzędzie będzie) i TOŻSAMOŚĆ (po czym poznać, że to, co wyszło zza tęczówki, to to samo narzędzie).
Tokmakov i in. 2021 (10.1109/ICCV48922.2021.01068) trzymają oba w pamięci rekurencyjnej i propagują położenie stałą
prędkością (+5,0 HOTA na KITTI Car); Tran i in. 2026 (10.48550/arXiv.2606.12601) rozdzielają jawnie stan tożsamości
(wolny, filtrowany przez GRU) od stanu wyglądu (szybki) i mierzą, że stan wyglądu ma 1,33 razy większą wariancję w czasie
niż stan tożsamości. To jest wyuczona wersja tego, co równanie Langevina i Koopman mierzą bez uczenia: składowa
`|λ| = 1` cząstek narzędzia to klucz do ponownej identyfikacji, składowa wolna to to, co filtr propaguje, składowa
szybka to to, co się odrzuca. Praktyczny zysk: tracker dostaje z tego samego dopasowania (a) model procesu dla
położenia, (b) wektor tożsamości do przypisania po zasłonięciu, (c) próg „to nie jest już to samo narzędzie", gdy
innowacja w składowej niezmienniczej przekroczy podłogę z permutacji.

### 4.3. Co równanie mówi o błędach detektora

`rozklad_bledu` pozwala rozdzielić chybienia detektora na te, które historia toru przewiduje (błąd detektora: ramka
łamie fizykę toru w sposób przewidywalny) i te, w których narzędzie naprawdę skoczyło. Pomiar (dokument 38, 4c):
sama innowacja wskazuje chybienia z AUC 0,60 do 0,68, pewność detektora 0,81 do 0,83. Jako samodzielny wykrywacz
błędów innowacja przegrywa; jako składnik obok pewności nie jest zmierzona. Uczciwa rola: nie zastępować pewności,
tylko wskazywać chybienia, przy których detektor jest pewny siebie (dokument 39, 5c: przy chybieniu wynik 0,80 wobec
0,89 przy trafieniu, więc pewność sama tego nie łapie).

## 5. Protokół: jak zdiagnozować, które cząstki niosą tożsamość narzędzia

Dane dostępne dziś: tokeny DINOv2 ośmiu klipów (lokalnie, CPU) i pooling latentu RF-DETR na 4800 klatkach. Tokeny
RF-DETR per patch wymagają jednej zmiany w `detector_latent.py` (zapis mapy cech `feat` zamiast tylko jej poolingu) i
przebiegu na Edenie, bo lokalnie nie ma `rfdetr`. Etykiety (tory GT) czytane WYŁĄCZNIE do oceny, po podziale na grupy
operacji jak w dokumentach 36 do 39. Kryteria zapisane PRZED wynikiem.

| Krok | Co | Miara | Kryterium |
|---|---|---|---|
| 0 | test rejestrów T6a na tokenach każdej klatki | norma przed normalizacją, stałość indeksu, związek z treścią | tokeny odrzucone raportowane; przy ViT-S spodziewane zero |
| 1 | równanie `D + JP + N` per token (adres) i per grupa tokenów: w ramce GT, w ramce detektora, scena, brzeg | skill, zysk pamięci przy RÓWNEJ pojemności (M4 kontra M2 z planu 37), autokorelacja reszty na lagach 1 do 4, kurtoza, podłoga z permutacji | grupa narzędzia różni się od sceny co najmniej o 10 punktów skill albo o 0,1 autokorelacji; inaczej podział tokenów nie ma dynamicznego sensu |
| 2 | test domknięcia: dopasowanie na samych tokenach narzędzia kontra narzędzie plus scena kontra narzędzie plus tokeny sąsiednie | spadek energii `JP` i spadek koloru reszty po dołożeniu | tokeny narzędzia są zamknięte, gdy dołożenie sceny nie zmniejsza `JP` ani nie wybiela reszty (`mz_zero_iff_warunkowa`); test Chapmana–Kołmogorowa na kilku krokach (Noé i in. 2013, 10.1063/1.4828816: skale czasu niezależne od lagu tylko dla procesu markowowskiego) |
| 3 | przypisanie po treści: prognoza `z^_{t+1\|t}` każdego tokenu narzędzia dopasowana węgiersko do tokenów klatki `t+1` | zgodność przypisania z ruchem ramki GT; udział tokenów, które „zgubiły" trajektorię | przypisanie po prognozie bije przypisanie po samym adresie (T32 mówi, że adres paruje populacje; narzędzie tego łamie) |
| 4 | Koopman na tokenach narzędzia: mody wolne kontra PCA16 kontra 16 losowych rzutów kontra permutacja | R² wolnych modów do opisu wyglądu (deskryptor z `koopman_tool.py`), położenia, prędkości | mody `\|λ\| ≈ 1` niosą wygląd, mody `0,98` położenie, gasnące prędkość; przewaga nad PCA i losowymi rzutami, jak zażądano w dokumencie 39, 5d |
| 5 | filtr `kalman_state.py` na stanie z tokenów (zamiast `box_pool`) | R² prędkości, białość i kurtoza innowacji | latent wnosi coś ponad ramkę tylko wtedy, gdy przebija filtr z ramek: korelacja 0,67, R² 0,44 (dokument 38, 4c) |

Falsyfikatory: (a) `JP ≈ 0` i biała reszta już na `box_pool` w 24 wymiarach: stan narzędzia jest zamknięty na poziomie
poolingu i tokeny nic nie dodają (zgodne z Gupta 2023, zysk pamięci znika przy 16 obserwablach); (b) grupa narzędzia
nierozróżnialna od sceny w kroku 1: podział tokenów nie ma dynamicznego znaczenia, a „tożsamość" jest własnością
całej klatki, jak w dokumencie 39, 5d; (c) krok 5 poniżej progu: równanie na tokenach jest diagnostyką, nie
składnikiem trackera.

Koszt: kroki 0, 1, 2, 4 na tokenach DINOv2 lokalnie, minuty na CPU (macierze 410 × 256 × 48 na klip); kroki 1 do 5
na tokenach RF-DETR: jeden przebieg `detector_latent.py` z zapisem mapy cech na Edenie, około 4800 klatek na
checkpoint, potem lokalnie.

## 6. Kontrargumenty

- **Swoboda cechowania (T0; Lusch, Kutz, Brunton 2018).** Każda odwracalna zmiana współrzędnych daje równoważne
  równanie; „wolne mody" i „jądro pamięci" są fizyczne tylko względem jawnie podanej metryki. Tu metryką jest projekcja
  losowa plus standaryzacja na foldach dopasowania (poprawka A1 z dokumentu 38, 4e, jeszcze niewdrożona).
- **Jedno narzędzie.** Tożsamości między narzędziami nie da się w tym zbiorze zmierzyć; składowa `|λ| = 1` będzie
  stałą i test jej treści wymaga zbioru wielonarzędziowego (dokument 36, sekcja 7).
- **Pamięć jest mała.** Na torach GT `JP` daje 1 do 2 punktów; „+4 punkty" z latentu (dokument 38, 4a) to różnica
  historii 3 i 8 kroków, nie kontrola bez historii. Równanie może wyjść markowowskie po dopisaniu prędkości i
  przyspieszenia, co byłoby wynikiem, nie porażką.
- **Próbkowanie.** Krok 5 klatek (tokeny DINOv2) i 3 klatki (latent detektora) to różne skale; Lin i in. 2023
  (10.1088/2632-072X/adbba8) pokazują, że zbyt rzadkie próbkowanie zmienia klasę równania. Wyniki z dwóch cache nie
  są porównywalne bez wspólnego kroku.
- **Grube ogony.** Innowacje mają kurtozę do 132; każda strata kwadratowa i każdy test białości bez odporności
  będzie zdominowany przez kilka skoków (grant: 1 procent cząstek niósł 80 procent wyniku łącznego). Wszystkie
  miary w protokole raportować z medianą po klatkach i po obcięciu 1 procenta największych kroków.
- **DMD już przegrał z różnicą klatek** (dokument 36, 3.1). Kryterium domknięcia jest inne niż „co się rusza",
  ale ciężar dowodu jest po stronie protokołu.

## 7. Literatura użyta w tym dokumencie

Mori–Zwanzig i pamięć: Lin, Tian, Anghel, Livescu 2021 (10.1137/21m1401759); Lin, Tian, Livescu 2022
(10.1137/22m1506146); Wang, Benner, Heiland 2026 (arXiv:2606.23341); Gouasmi, Parish, Duraisamy 2016
(10.1098/rspa.2017.0385); Curtis, Alford-Lago 2021 (10.1103/physreve.103.012201: reszta DMD równoważy człon pamięci;
DMD wystarcza tylko przy pełnej obserwacji); Gupta i in. 2023 (10.48550/arXiv.2310.10745); She i in. 2022
(10.1063/5.0130033); Lei, Baker, Li 2016 (10.1073/pnas.1609587113); Noé i in. 2013 (10.1063/1.4828816);
Lapolla, Godec 2021 (10.1103/PhysRevResearch.3.L022018: diagnostyka pamięci z modelami zerowymi i testem podpróbkowania).

Koopman i wolne cechy: Klus i in. 2017 (10.1007/s00332-017-9437-7); Williams, Kevrekidis, Rowley 2014
(10.1007/s00332-015-9258-5); Kamb i in. 2018 (10.1137/18m1216572: współrzędne opóźnień jako obserwable Koopmana,
Takens `N ≥ 2n + 1`); Wiskott, Sejnowski 2002 (10.1162/089976602317318938); Franzius, Wilbert, Wiskott 2011
(10.1162/NECO_a_00171). Związek SFA z TICA jest w literaturze, ale nie został tu pobrany; do zweryfikowania przed
cytowaniem.

Tokeny jako cząstki i rejestry: Geshkovski i in. 2023 (10.48550/arXiv.2312.10794: tokeny na sferze, uwaga jako
oddziaływanie, zbieganie do jednego skupienia); Darcet i in. 2023 (10.48550/arXiv.2309.16588); Lappe, Giese 2025
(10.52202/085713-0037).

Tożsamość w śledzeniu: Tokmakov i in. 2021 (10.1109/ICCV48922.2021.01068); Tran i in. 2026 (10.48550/arXiv.2606.12601);
Becker i in. 2019 (arXiv:1905.07357); Greenberg i in. 2021 (semanticscholar 680b6327, znany tylko abstrakt).

Próbkowanie i klasa równania: Lin i in. 2023 (10.1088/2632-072X/adbba8).

Wyszukiwania Undermind użyte bez nowego deep search: „Konstrukcyjne rozszerzanie stanu o mody pamięci",
„jak z samych trajektorii rozpoznać klasę równania ruchu cząstki" (workspace grantu), „operator Koopmana na latencie
wideo jako kryterium selekcji", „śledzenie końcówki narzędzia przez okluzje" (workspace ViT); osiem wyszukiwań
semantycznych i jedenaście odczytów PDF z 2026-09-22.

## 8. Data gaps

- Brak tokenów RF-DETR per patch: `detector_latent.py` zapisuje tylko pooling; potrzebna zmiana i przebieg na Edenie.
- Test rejestrów T6a nieuruchomiony ani w grancie, ani tutaj.
- Poprawka A1 (standaryzacja na foldach) niewdrożona; liczby skill z dokumentu 38 do powtórzenia.
- Krok próbkowania niejednolity między cache (5 i 3 klatki).
- Zbiór jednonarzędziowy: składowa tożsamości między narzędziami niemierzalna.
