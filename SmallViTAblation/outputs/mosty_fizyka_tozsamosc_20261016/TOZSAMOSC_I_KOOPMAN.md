# Tożsamość w latencie ViT: co dają niezmienniki, Koopman i obserwowalność

16 października 2026. Notatka badawcza na podstawie lokalnej bazy Grant i zweryfikowanych źródeł pierwotnych. **Nie wykonano nowych pomiarów modelu, treningów ani zmian aplikacji.** Pliki Grant były tylko czytane.

## Najważniejsza odpowiedź

Nie wiemy jeszcze, gdzie w tym ViT jest informacja o tożsamości. Najrozsądniejszy kandydat to **rozproszona podprzestrzeń reprezentacji, która rozróżnia wybraną tożsamość, zachowuje tę informację przy zmianie położenia/wyglądu i daje się oddzielić od kontekstu nagrania**. Wolne mody Koopmana mogą pomóc wytypować taką podprzestrzeń. Nie identyfikują jej automatycznie.

„Gdzie” powinno znaczyć: **w którym bloku, przed/po której operacji, w jakiej przestrzeni i podprzestrzeni, dla jakich tokenów oraz przy jakim zakresie danych**. Nie musi istnieć jeden „neuron narzędzia” ani jedna studnia przypisana kategorii. Odczytywalność informacji, jej trwałość oraz korzystanie z niej przez detektor są trzema różnymi twierdzeniami.

W naszym modelu jest 600 tokenów, sześć bloków, residual 192D i ukryte aktywacje GELU 768D. Głowa ma pięć wyjść na token: logit środka oraz cztery wielkości regresji ramki. **Nie ma jawnego wyjścia klasy rodzaju narzędzia.** Nawet idealna lokalizacja bbox nie dowodzi, że sieć rozróżnia rodzaje narzędzi albo śledzi fizyczny egzemplarz.

## 1. Najpierw ustalić, jakiej tożsamości szukamy

| Cel | Wymagane etykiety i kontrast | Co wolno stwierdzić |
|---|---|---|
| Obiekt/narzędzie wobec tła | Maska narzędzia lub zweryfikowany region; tło z tych samych obrazów | Sygnał obiektowości/treści regionu; bbox sam nie jest segmentacją |
| Kategoria narzędzia | Co najmniej dwie kategorie w wielu niezależnych filmach; nowe egzemplarze tej samej kategorii | Informacja o rodzaju, niezależna od części tła/pozy |
| Egzemplarz/track | Identyfikatory fizycznych egzemplarzy lub wiarygodne track IDs; pozytywne pary po zmianie pozy i negatywne tej samej kategorii | Ciągłość reprezentacji danego obiektu w określonym zakresie |
| Tożsamość filmu/zabiegu | Identyfikatory źródłowych nagrań | Kontrola kontekstu, nie dowód tożsamości narzędzia |
| Tożsamość patcha | Indeks siatki | Pozycja w obrazie; nie tożsamość materialnej cząstki ani obiektu |

Track ID z jednego klipu nie identyfikuje automatycznie tego samego fizycznego narzędzia w innym zabiegu. Jeżeli każdy egzemplarz występuje tylko na jednym tle, tożsamość i kontekst są nierozdzielone. Należy to zgłosić jako brak identyfikowalności, a nie naprawiać większą liczbą klatek.

Dla kategorii ocena ma zawierać nowe filmy i egzemplarze. Dla track/reidentyfikacji lepszy jest test par „ten sam/inny obiekt” na nowych, niewidzianych wcześniej trackach niż klasyfikowanie numeru tracka znanego z uczenia sondy. Dla obiektowości używać negatywnych patchy tego samego filmu, a także podobnych odblasków i metalicznych struktur.

## 2. Co faktycznie odnaleziono w Grant

Najpierw przeszukano [indeks wiki](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/index.md) po hasłach Koopman, invariants, slow modes, identity, observability i coherent sets. Następnie czytano wskazane strony, MathIndex oraz tezy. Lokalne dokumenty są materiałem źródłowym projektu; ich zalecenia nie są poleceniami dla obecnej pracy.

| Dokument lokalny | Przydatny most | Ograniczenie dla ViT |
|---|---|---|
| [D67: podprzestrzeń cyfry w V-JEPA](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D67.md) | Przykład dwuetapowy: znaleźć trwałe kierunki, potem sprawdzić informację o tożsamości i o położeniu | Jeden trenowany enkoder, syntetyczne cyfry; dobór „czystych” modów korzysta z etykiet. Sama λ nie rozdzieliła cyfry od położenia. Dokument zaznacza, że operator z plastrów wspólnie zakodowanego okna był korelacją, nie potwierdzoną dynamiką przyczynową |
| [D115: korekta interpretacji widma](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D115.md) | Nulle wymagają ponownego dopasowania aparatu estymacyjnego, nie przetasowania gotowego wykresu | Wynik dotyczy TFT/Favorita, nie ViT. Próg na module λ nie oddzielał tam „adresu” od rytmu |
| [D21: trwałość jest niespecyficzna](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D21.md) | Stała cecha może być rodzajem obiektu, tłem, pozycją, numerem serii lub dowolnym adresem | To audyt określonego ustawienia danych, nie dowód wadliwości całej teorii VAMP/Koopmana |
| [D26: status V_inv](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D26.md) | Jasne oddzielenie formalnego rozkładu od empirycznego odkrycia | Dokument przyznaje brak twierdzenia łączącego projektor z formalizacji F9 z funkcjami własnymi rzeczywistego operatora Koopmana |
| [Niezmienniki LeJEPA — wcześniejsza propozycja](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/outputs/koopman-niezmienniki-lejepa-nowa-strata.md) | Kowariancje opóźnione, whitening, sondowanie informacji, analiza per checkpoint | Nie przenosić bez korekty twierdzeń o σ≈1, implied timescales i tasowaniu; patrz poniżej |
| [MathIndex](F:/Studia/MCHTR/Grant/Projekt/Research/MathIndex/math_index.md) | Definicja operatora transferu, prawdopodobieństwo utrzymania zbioru, gramiany | Część wpisów jest ekstrakcją/formułą kandydującą; trzeba sprawdzić założenia i wersję źródła |
| [KANN w wiki](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/sources/naiman-azencot-2023-koopman-neural-network-analysis.md) | Operatorowa interpretacja stanów sieci już istnieje w literaturze | Model sekwencyjny nie jest automatycznie równoważny sześciu różnym blokom naszego ViT |

Wpisy [observability](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/concepts/observability.md) i [coherent sets](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/concepts/coherent-sets.md) są oznaczone jako stuby. Nie traktowano ich jako dowodów.

**Korekty potrzebne przy wykorzystaniu tej bazy, bez edytowania Grant:**

- SVD surowego C01 nie jest pełną estymacją VAMP: potrzebne są metryki obu rozkładów i whitening. W ogólności C00≠C11.
- Wysoka singular value oznacza silną przewidywalną relację między dwiema funkcjami, niekoniecznie stałość jednej funkcji. Obrót może zachowywać normę i mieć singular values równe 1.
- Prawdziwa niezmienna tożsamość **nie musi zniknąć po tasowaniu czasu wewnątrz tego samego tracka**. Taki null zachowuje tożsamość i statyczne tło. Muszą istnieć osobne kontrole dla dynamiki oraz dla semantyki.
- Nie zamieniać dowolnego σ na fizyczny czas relaksacji. Standardowy odczyt −τ/log|λ| dotyczy odpowiedniego modelu własnego i jego założeń; σ ogólnego niestacjonarnego operatora nie jest automatycznie takim λ.
- Nie zakładać detailed balance tylko po to, żeby uzyskać rzeczywiste widmo. Symetryzacja zmienia estymowany problem.

## 3. Trzy osie, których nie wolno utożsamić

1. **Czas nagrania s:** zamrożone wagi θ, rzeczywiste obrazy I_s. Obiekt może się przemieszczać. Ten sam indeks patcha obserwuje inne części sceny; do tożsamości potrzebne są korespondencje obiektu/tracka.
2. **Głębokość l:** jeden obraz, zamrożone θ, różne mapy attention/MLP. Nie ma podstaw, by sześć różnych bloków zastąpić jedną iterowaną macierzą K i odczytać z jej potęg „czas życia” tożsamości.
3. **Krok uczenia k:** ten sam obraz badany w θ_k. Zmienia się sam enkoder i często bufory. Stałość reprezentacji między checkpointami nie oznacza tożsamości obiektu w czasie fizycznym. Analizę tożsamości wykonuje się osobno w kolejnych stanach; ewentualne wyrównanie baz dopasowuje wyłącznie na niezależnych danych discovery.

**Oś augmentacji** jest czwartym, eksperymentalnym porządkiem: I→gI. Operator losowych augmentacji jest prawidłowo zdefiniowanym operatorem warunkowego uśredniania, ale jego „czas” i niezmienniki wynikają z wybranej dystrybucji g. Nie są automatycznie prawem ruchu filmu ani treningu.

Dla zamrożonego enkodera obserwacje h_s mogą nie być stanem Markowa. Ocena z jednego tokenu nie zamyka dynamiki całego obrazu, zwłaszcza po attention. Dlatego pierwszy opis powinien brzmieć „operator predykcji obserwabli z par danych”; mocniejsza interpretacja dynamiki wymaga testu domknięcia i wielokrokowej predykcji.

## 4. Trzy konkretne testy

Wszystkie są planami. Nie wymagają zmiany funkcji straty ani dodawania własności światła do aplikacji. Kosztowne wykonanie można później umieścić na Edenie.

### I1. Atlas informacji: co jest czytelne i odporne na zmianę kontekstu?

**Pytanie.** W którym z sześciu bloków i w residual192/GELU768 można odczytać obiektowość, kategorię lub zgodność tracka ponad informację o położeniu, tle i filmie?

**Dane i jednostka.** Ustalić dostępność kategorii, track IDs i masek; obecnych bbox nie przemianowywać na etykiety rodzaju narzędzia. Pilot: co najmniej dwie kategorie obecne w kilku niezależnych filmach każda oraz wiele wystąpień/egzemplarzy na kategorię. Liczba zależy od rzeczywistych adnotacji. Jeżeli kategorii lub tracków nie ma, wykonać wyłącznie obiektowość albo przygotować małą zweryfikowaną adnotację — bez udawania, że znamy identity. Niezależne jednostki to film/zabieg i seed modelu, nie patch ani klatka.

**Reprezentacja.** Dla punktu l pobrać h_{s,p}^{(l)}∈R^192 i z_{s,p}^{(l)}∈R^768. Porównać tokeny w zweryfikowanej masce, okolice granicy, tło i pooling obiektu r_s=Σ_p w_{s,p}h_{s,p}, Σw=1. Pooling z GT jest oracle diagnostycznym; dostęp do maski/GT musi być identyczny w kontrolach. Podać token-level i pooled wynik, bo średnia może ukryć rozproszony kod. Token po attention nie jest czystą lokalną próbką obiektu.

**Wzory i procedura.** Z par tego samego obiektu pod dozwolonymi zmianami g:

C=Cov(r),  D=E[(r(gI)−r(I))(r(gI)−r(I))^T].

Kandydackie kierunki stabilne względem tej rodziny zmian minimalizują

a^T D a / a^T(C+εI)a.

Rozwiązać na discovery, sprawdzić na nowych filmach. To uogólniony test niezmienności/slow-feature, **nie dowód funkcji własnej dynamiki filmu**. Na stałym budżecie d∈{4,8,16,32} porównać taki rzut, PCA z nieuśrednionych klatek, losową podprzestrzeń i pełny latent. Następnie niezależnie trenować regularizowane sondy dla kategorii, obiektowości i nuisance: pozycja, skala, jasność, film. Regułę wyboru d/warstwy i selekcji „identity minus nuisance” zamrozić na discovery. Jeśli etykiety uczestniczą w wyborze podprzestrzeni, nazywać ją nadzorowaną selekcją.

Dla tracka zamiast klasyfikowania znanego numeru: weryfikacja par/retrieval na niewidzianych trackach, hard negatives tej samej kategorii i podobnej pozy; gallery/query oddzielone w czasie. Reidentyfikacja fizycznego egzemplarza poza filmem wymaga rzeczywistych etykiet tej relacji.

**Kontrole przecieku.** Cały film, wszystkie augmentacje obrazu i ciągłe fragmenty tracka trafiają do jednego splitu. Dla kategorii odłożyć egzemplarze. Sondy metadanych, samego bbox i tła ujawniają shortcut. Filtry, whitening, PCA, pooling policy, wybór modów i hiperparametry dopasować bez testu. Kontrola init ma ten sam stem i pozycje; odczytywalność samej pozycji nie jest dowodem nauczenia semantyki.

**Endpoint i falsyfikacja.** Balanced accuracy/AUPRC dla kategorii/obiektowości, retrieval i same/different performance dla tracka, przy przewadze ponad tło/pozycję i z raportem niepewności po filmach. Sukces tylko na losowo dzielonych klatkach, zanik po zmianie tła lub brak przewagi nad metadanymi falsyfikuje deklarowaną semantyczną lokalizację. Brak liniowego odczytu nie dowodzi całkowitego braku informacji; ograniczony nieliniowy probe jest kontrolą o jawnym dodatkowym koszcie.

**Osie.** Najpierw jeden zamrożony checkpoint, wszystkie bloki; dopiero później init i 2–3 ustalone etapy treningu. Nie wybierać warstwy i checkpointu według końcowego testu.

**Wynik, którego szukamy.** „W bloku l, rzut r wymiarów zachowuje informację o kategorii na nowych filmach mimo zmiany pozy i tła” — a nie „neuron c jest narzędziem”.

### I2. Czy to właśnie trwałe mody i koherentne grupy niosą tę informację?

**Pytanie.** Czy kierunki dobre w I1 są również trwałe względem rzeczywistego czasu nagrania lub jawnie wybranego operatora augmentacji, i czy ta trwałość coś wnosi ponad statyczną tożsamość filmu?

**Dane.** Zamrożone wagi i długie, oznaczone tracki; r_s z tego samego materialnego obiektu, a nie stałego indeksu patcha. Pary nie przekraczają cięcia montażowego, końca tracka ani zmiany obiektu. Na początek 3 lagi określone w sekundach/klatkach, znane fps i 2 rozłączne zakresy czasowe. Gdy są tylko pojedyncze obrazy, wykonać wersję augmentacyjną i tak ją nazwać; nie estymować „dynamiki wideo” z checkpointów.

**Operator i wzory.** Dla procesu o odpowiednim stanie:

(K_τ f)(x)=E[f(X_{s+τ}) | X_s=x].

Na cechach r z par danych, po ustalonym centrowaniu i regularyzacji:

C00=E[r_s r_s^T], C01=E[r_s r_{s+τ}^T], C11=E[r_{s+τ}r_{s+τ}^T],  
T_τ=(C00+εI)^(−1/2) C01 (C11+εI)^(−1/2).

SVD T_τ daje kierunki predykcyjne z lewej i prawej strony. To nie musi być jedna wspólna podprzestrzeń. Stałą obserwablę wydzielić; nie liczyć jej jako odkrytej identity. Dla kandydackiej **tej samej** funkcji φ mierzyć bezpośrednio:

E_inv(φ)=E[|φ(r_{s+τ})−φ(r_s)|²]/Var[φ(r_s)],

oraz stabilność na kilku lagach i semantykę z I1. Małe E_inv bez rozróżniania tożsamości to zwykła stałość, nie rozwiązanie problemu. Koopman przewiduje ewolucję obserwabli; jego podejście nie wymaga, aby surowe współrzędne r miały zamkniętą liniową dynamikę.

Dla dopasowanej mapy predykcyjnej A_τ sprawdzić na odłożonych sekwencjach r_{s+τ}≈A_τ r_s i A_{2τ}≈A_τ² w uzgodnionej, wspólnej bazie. Nie składać bezpośrednio dwóch różnych wybielonych TCCA map, których lewa/prawa baza się różni. Brak zgodności jest informacją o braku domknięcia, pamięci lub niestacjonarności. Rozszerzenie o opóźnienia może pomóc, ale zwiększa wymiar i wymaga nowej walidacji.

**Koherentne zbiory jako analiza pomocnicza w tym samym teście.** Dla zbiorów A_s,A_{s+τ} w przestrzeni reprezentacji:

ρ=P(r_{s+τ}∈A_{s+τ} | r_s∈A_s).

Zbiory dobierać na discovery i oceniać na teście przy ustalonych masach/udziałach, z porównaniem do szansy P(r_{s+τ}∈A_{s+τ}). Cała przestrzeń daje ρ=1 i jest bezużyteczna. Koherencja mówi o małej wymianie populacji; nie określa, czy granica oddziela kategorie, tracki czy nagrania. Osobno ocenić czystość semantyczną oraz kontekst. Nie nazywać klastrów 600 tokenów pojedynczego obrazu „materialnymi wirami”.

**Dwa różne nulle.**

- Tasowanie kolejności **wewnątrz tracka**, z ponownym dopasowaniem całego operatora i rankingu, testuje dodatkową informację o kolejności ponad stałe cechy tracka. Nie oczekiwać zaniku prawdziwej identity.
- Rozrywanie par **między trackami**, z dopasowaniem kategorii/pozy/tła zależnie od testowanego poziomu identity, niszczy właściwą relację tożsamości. Dla kategorii trzeba również mieć kontrast między kategoriami; null „inna instancja tej samej kategorii” zachowuje kategorię celowo.

Dodatkowo kontrola persistence r_{s+τ}=r_s, mean-per-track i model pozy/tła. Dla diagnostyki czasu porównać surowe cechy z resztą po odjęciu trackowej średniej; ta operacja **celowo usuwa stałą tożsamość** i nie może być jedynym pomiarem jej obecności. Średnie z całego testowego tracka wykorzystane opisowo nie mogą trafiać do prospektywnego predyktora; wersja predykcyjna korzysta tylko z przeszłości.

**Endpoint i falsyfikacja.** Held-out błąd predykcji, E_inv, odczyt etykiety przy stałej liczbie modów oraz przyrost ponad I1/PCA/random; koherencja ponad dopasowaną bazową częstość. Identity czytelna wyłącznie jako numer filmu, brak przewagi wolnych modów albo wyniki identyczne z refit null nie potwierdzają preferencyjnego kodowania identity w „wolnej części”. Informacja o tożsamości może nadal istnieć gdzie indziej. Nie wybierać progów λ po etykietach testowych.

**Osie.** Czas filmu lub augmentacji, osobno dla każdego l,k. W treningu śledzić wynik tej procedury, nie sklejać różnych θ_k w jedną stacjonarną trajektorię Koopmana.

**Interpretacja fizyczna.** Możemy otrzymać populację reprezentacji zachowującą określoną informację podczas zmiany obrazu. To sensowny most do trwałych obserwabli i słabo mieszających zbiorów. Nie jest to dowód optycznego potencjału ani kolloidowej pułapki.

### I3. Informacja odczytywalna kontra używana: obserwowalność i nietautologiczny patching

**Pytanie.** Czy mała podprzestrzeń identity znaleziona w I1/I2 tylko pozwala zewnętrznej sondzie odczytać etykietę, czy także przyczynowo uczestniczy w aktualnej detekcji?

**Dane.** Pary dawców o rozdzielonych czynnikach: ta sama kategoria, inna poza/tło; inna kategoria, podobna poza/tło; ten sam/inny egzemplarz tej samej kategorii, jeśli etykiety pozwalają. Dla obiektowości: obiekt/tło przy dopasowanej pozycji i jasności. Nie zastępować tego losowym dawcą o kompletnie innym obrazie.

**Wzór obserwowalności.** Dla ustalonego tokenu i l niech G_l=∂y/∂h_l oznacza pochodną **całego suffixu** do wybranych ciągłych wyjść; normalizację ich jednostek ustalić z discovery. Lokalna czułość na zmianę δh to ||G_l δh||²=δh^T G_l^T G_l δh. Wartość tr(P G_l^T G_l P) ocenia widoczność podprzestrzeni P w tym odczycie, nie jej semantykę.

Pojedynczy odczyt pięciu liczb ma lokalną rangę nie większą niż 5. Jądro tego Jacobianu nie dowodzi braku informacji ani skończonej/globalnej nieistotności. Przy wcześniejszych blokach i wszystkich kandydatach trzeba policzyć pełny downstream, a nie założyć takiego samego ograniczenia dla całego obrazu.

Dla hipotetycznego zwalidowanego modelu liniowego r_{s+1}=A r_s, y_s=C r_s istnieje skończony gramian:

W_o(H)=Σ_{j=0}^{H−1}(A^j)^T C^T C A^j.

Dotyczy obserwowania **tego samego stanu przez rzeczywiste kolejne pomiary**. Nie używać go z potęgami jednego MLP do sześciu różnych bloków ani interpretować wyjść z przyszłych checkpointów jako pomiarów nieruchomego stanu. W ViT najpierw wystarczy lokalny G_l i dokładna interwencja; gramian jest dalszą możliwością po zaliczeniu domknięcia I2.

**Dokładna interwencja.** Dla wcześniej zamrożonej ortogonalnej podprzestrzeni P w surowym residualu:

h'=h + αP(h_donor−h),  α∈{0,0.5,1}.

Jeśli P wyznaczono po whiteningu u=L^(−1)(h−μ), prawidłowa mapa to h'=μ+L[u+αP(u_d−u)]; nie podstawiać wybielonego P bezpośrednio do surowych współrzędnych. Dla interwencji GELU analogicznie 768D, z ponownym W2/residual i resztą sieci. Wybór P, tokenów, rangi oraz donor policy wyłącznie discovery. Nie dobierać kierunku z gradientu właściwej zmiany wyjścia w przykładzie testowym.

**Nietautologiczne przewidywanie.** Przed testem określić, co powinno się zmienić, a co pozostać stałe. Dla kategorii/tracka ocenić także niezależny od selekcji probe, ale sama zmiana jego wyniku po transplantacji danych, z których tę sondę zbudowano, jest słabym dowodem. Kluczowe są pełne skutki na modelu, selektywność dawców oraz noising/denoising. W przypadku głowy bez rodzaju narzędzia zmiana kategorii przy identycznym bbox może prawidłowo nie zmienić wyjścia: wtedy można wykazać zakodowanie kategorii, ale nie jej używanie przez natywny klasyfikator, którego nie ma.

Dla postulowanej obiektowości przewidywać efekt na heat i lokalizację przy zachowaniu niezmienianych aspektów. Dla hipotezy „identity pomaga lokalizować” oczekiwany efekt powinien wystąpić w określonych trudnych warunkach, np. częściowa okluzja, oraz zostać odtworzony przez dawcę zachowującego tę informację mimo zmiany tła. Przywrócenie całego końcowego residualu albo dokładne cofnięcie własnej korekty to tylko kontrola dodatnia.

**Kontrole.** Rank/norm-matched random P, PCA, wybrana nuisance podprzestrzeń, identyczny donor, niewłaściwa identity przy podobnym kontekście; ta sama identity przy innym kontekście. Oddzielić oryginalny model i zmienione stany poza rozkładem; wielkość zmiany i odległość od naturalnych aktywacji raportować. Rescue nie jest samoistnym dowodem normalnego używania kierunku — kontrolować uśpione drogi i rzeczywiste usunięcie sygnału na naturalnych danych.

**Dokładny null dla 768D.** Dla z po GELU i W2∈R^(192×768), dim ker(W2)≥576. Dowolna interwencja z'=z+v z v∈ker(W2), przy zamrożonym pozostałym stanie, nie zmienia W2z+b2, residualu ani całego downstream; w obliczeniach numerycznych wymaga kontroli tolerancji. Jest to szczególnie mocna kontrola poprawności hooka i przykład, dlaczego odczytywalność z 768D nie oznacza użycia przez detektor. Rozdzielić z_row=W2†W2z oraz z_null=(I−W2†W2)z, a sondy I1 ocenić na obu przy porównywalnym budżecie. Informacja czytelna w z_null może być redundantnie obecna także w z_row; sam dobry probe w jądrze nie dowodzi, że ta sama informacja jest całkowicie niewykorzystywana przez sieć. Kierunki z ker(W2) są poprawnymi interwencjami na węźle, lecz nie muszą być osiągalne zmianą obrazu lub wejścia LN: lokalny obraz mapy x→GELU(W1 LN(x)+b1) ma rangę nie większą niż 191 przy residual192 i symetrii przesunięcia LN. Nie utożsamiać wysokiego wymiaru algebraicznego jądra z liczbą niezależnych naturalnych zmian semantycznych. Ten null należy do I3, nie jest czwartym testem.

**Endpoint.** Oddzielnie: odczyt etykiety z reprezentacji; fixed-candidate logit/ltrb/krawędzie; pełny peak-mask/top10 dekoder, IoU i przypisanie obiektu. Dla translacji zachować aktualne reguły osiągalności kotwicy z PROTOKOL.md. Zmiana confidence nie równa się zmianie kategorii ani poprawności bbox.

**Falsyfikacja.** Patching działa równie dobrze dla niewłaściwych dawców/random P; wpływa przede wszystkim na pozycję/tło; albo rescue istnieje tylko dla silnie nienaturalnych stanów — brak dowodu specyficznego mechanizmu identity. Dobry probe przy braku efektu na detektorze uzasadnia węższy wynik: informacja jest odczytywalna, lecz jej rola w badanym zadaniu nie została potwierdzona. Mały liniowy gradient wymaga kontroli skończonej perturbacji i redundancji, nie deklaracji „sieć tego nie używa”.

**Osie i minimum.** Jeden zamrożony checkpoint, najwyżej 1–2 warstwy wybrane na discovery, nowi dawcy i filmy; później niezależne seedy. Nie trzeba najpierw liczyć wszystkich par i pełnego atlasu 806 checkpointów.

## 5. Co byłoby wiarygodnym wynikiem publikacyjnym?

Najbardziej wartościowy wynik ma postać:

> W określonych blokach ViT istnieje niskowymiarowa, przenoszalna reprezentacja konkretnego poziomu tożsamości. Jest bardziej odporna na wskazane nuisance niż kontrolne kierunki. Rozdzielenie semantyki i kontekstu potwierdzono na nowych filmach/egzemplarzach, a interwencje pokazują jej określoną rolę — lub brak wykrywalnej roli — w lokalizacji.

Most do fizyki jest wtedy precyzyjny: **trwała obserwabla plus obserwowalność funkcjonalna**, ewentualnie koherentny zbiór reprezentacji. Nie trzeba nazywać tego światłem, siłą optyczną ani studnią energii. Jeżeli slow modes nie przewyższą statycznych metod I1, warto opublikować rzeczywiste rozdzielenie identity/nuisance, zamiast utrzymywać Koopmana w głównej tezie wyłącznie dla narracji.

Minimalny kolejny krok to audyt etykiet i I1 na istniejącym modelu. I2 potrzebuje rzeczywistych par czasu/augmentacji, I3 — zamrożonej hipotezy i niezależnych dawców. Jedno zdjęcie z wieloma checkpointami wystarcza do oglądania rozwoju reprezentacji, ale nie identyfikuje, która informacja jest kategorią, egzemplarzem albo kontekstem.

## 6. Źródła pierwotne zweryfikowane w tej notatce

1. **Wu, H., & Noé, F. (2017; wersja v3 2019).** *Variational approach for learning Markov processes from time series data.* [Pełny tekst](https://arxiv.org/html/1707.04659). Sprawdzono definicję operatora, feature TCCA, whitening i walidację. VAMP obsługuje także procesy nieodwracalne i niestacjonarne; nie oznacza to automatycznej identyfikowalności identity w dowolnych danych ViT. Wzory estymacyjne w I2 są zastosowaniem tej metody.
2. **Noé, F., & Nüske, F. (2013).** *A Variational Approach to Modeling Slow Processes in Stochastic Dynamical Systems.* [Wydawca](https://epubs.siam.org/doi/10.1137/110858616), [preprint](https://arxiv.org/abs/1211.7103). Sprawdzono abstrakt i zakres: wariacyjna aproksymacja wolnych procesów z obserwabli opóźnionych. Nie daje twierdzenia, że wolność równa się tożsamości semantycznej.
3. **Froyland, G., Lloyd, S., & Santitissadeekorn, N. (2010).** *Coherent sets for nonautonomous dynamical systems.* [Tekst autorów](https://web.maths.unsw.edu.au/~froyland/FLS.pdf), [arXiv](https://arxiv.org/abs/0911.0717). Zweryfikowano zakres definicji: słabo mieszające, zmienne w czasie obszary przestrzeni stanów. Kryteria masy/transportu nie nadają grupie etykiety „narzędzie”.
4. **Yeung, E., Liu, Z., & Hodas, N. O. (2017 preprint; ACC 2018).** *A Koopman Operator Approach for Computing and Balancing Gramians for Discrete Time Nonlinear Systems.* [PDF autora](https://zhiyuan-home.github.io/pubs/koopman.pdf), [arXiv](https://arxiv.org/abs/1709.08712). Sprawdzono konstrukcję gramianu obserwowalności i jej założenia. I3 używa najpierw prostszej lokalnej czułości rzeczywistego suffixu, nie zakłada globalnej dokładnej realizacji Koopmana.
5. **Naiman, I., & Azencot, O. (2023).** *An Operator Theoretic Approach for Analyzing Sequence Neural Networks.* AAAI 37(8), 9268–9276. [Oficjalna publikacja](https://ojs.aaai.org/index.php/AAAI/article/view/26111). Zweryfikowano oficjalny abstrakt i metadane. KANN stanowi prior art operatorowej interpretacji sieci sekwencyjnych; semantyka widma z ich przykładów nie jest pomiarem naszego ViT.
6. **Makelov, A., Lange, G., & Nanda, N. (2023).** *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching.* [Pełny tekst v2](https://arxiv.org/html/2311.17030v2). Zweryfikowano problem skutecznego steeringu przez nietypową/uśpioną drogę. Dlatego I3 oddziela odczyt sondy, sterowanie oraz rolę w naturalnej detekcji.

Dokumenty Grant stanowią inspirację i historię hipotez wraz z ich korektami. Nie przeliczano ich artefaktów i nie audytowano ponownie wszystkich historycznych liczb; żadnej z nich nie uznano za wynik SmallViT.

