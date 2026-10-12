# Układ wejście-stan-wyjście i równanie prognozy

## Jaki to układ

Pełny stan uczenia obejmuje parametry sieci θ, pierwsze i drugie momenty AdamW, bufory BatchNorm, krok i harmonogram lr, wersję zbioru oraz stan kolejki próbkowania. Aktualizacja jest nieliniową, dyskretną dynamiką z losowymi paczkami i augmentacjami. Harmonogram lr oraz zmiana zbioru powodują nieustacjonarność. Deterministyczny zapis można uzyskać dopiero po włączeniu stanu generatorów do pełnego stanu; to nie uprawnia prognozy do odtwarzania rzeczywistych przyszłych paczek.

Wejście fizyczne to wykonana paczka B_k, zrealizowane augmentacje i związana z nią aktualizacja. Mierzony wynik to `s_k=((z_k-mu)/sd)·w` stałej sondy, z kalibracją zamrożoną w kroku 1000. Nasz identyfikowany stan to opóźniona historia tego wyniku oraz historyczne podsumowania wpływów. Nie jest to pełny stan Markowa sieci. Dwie różne sieci mogą mieć to samo s i inny przyszły ruch.

`v_k=s_k-s_(k-1)` jest przyrostem latentu. Pęd AdamW jest pamięcią gradientów i ma inne jednostki oraz dynamikę; jego zapisane pchnięcie na s jest osobną cechą. Przyspieszenie `v_k-v_(k-1)` jest kombinacją już dostępnych opóźnień i nie powiększa informacji, gdy oba są w modelu. Przy niezależnym błędzie pomiaru stanu o wariancji σ² pierwsza różnica ma wariancję błędu 2σ², druga 6σ². Rzeczywista telemetria float16 nie musi spełniać założenia niezależności; nie stosujemy tych liczb jako pomiaru jej szumu.

## Lokalna linearyzacja, lifting i opóźnienia

Lokalne rozwinięcie aktualizacji wokół obecnego stanu daje macierze pochodnych i wejście, ale dokładność zależy od małego odchylenia oraz punktu pracy. Macierze zmieniają się wraz z θ i polityką; 500 iteracji może wyjść z tego zakresu.

Koopman działa liniowo na funkcjach pełnego stanu. Skończony zbiór obserwabli jest dokładny tylko przy domknięciu odpowiedniej podprzestrzeni. DMDc i lifting kwadratowy są tutaj estymowanymi aproksymacjami, a nie gwarancją globalnej linearyzacji dowolnej sieci. [Źródła i ograniczenia](zrodla/LITERATURA.md).

Opóźnienia są praktyczną aproksymacją brakującego stanu. Dla skończonej liniowej dynamiki obserwowanego x i ukrytego h ich eliminacja daje dokładnie dryf A, pamięć `B E^(j-1) C`, pamięć wejścia `B E^(j-1) F` i propagowany ukryty warunek początkowy. [Wyprowadzenie oraz zakres Lean/SymPy](matematyka/MATEMATYKA.md). Ten przykład tłumaczy wspólne występowanie Koopmanowskiego dryfu, jądra i kolorowych reszt. Nie wyznacza tych macierzy dla ViT.

## Postać estymowana bez przyszłych paczek

Równanie proponowane i przetestowane jako regresja bezpośrednia:

```text
d_H(t) = s_(t+H) - s_t
v10_(t-j) = s_(t-10j) - s_(t-10(j+1)),  j=0..19
x_t = s_t - mean(s_t,s_(t-10),...,s_(t-490))

dhat_H(t) = sum_j K_(H,j) v10_(t-j) - kappa_H x_t
             + sum_l D_(H,l) mean_l(pchniecie_paczki_do_t)
             + sum_l E_(H,l) mean_l(pchniecie_AdamW_do_t)
             + sum_l G_(H,l) mean_l(opis_paczki_do_t)

l in {10,50,100} wykonanych aktualizacji.
```

Każde pchnięcie i opis kończą się na ostatniej wykonanej paczce t-1 w indeksowaniu plików. Opis zawiera projekcję i rozrzut reprezentacji narzędzia w paczce, projekcję całego obrazu, udział obrazów z narzędziem, średni kontrast/jasność/odbicie, stratę, normę gradientu i lr. Sondy nie zastępują tych obrazów. Identyfikatory paczki i pełne augmentacje pozostają w oryginalnym nagraniu do audytu; predyktor korzysta z opisanych agregatów, nie z one-hot wszystkich identyfikatorów.

Współczynniki wspólne dla sond estymujemy ridge na dojrzałych parach r,r+H, z końcem etykiety przed odcięciem. Standaryzacja, wybór ridge i okna następują na treningu/walidacji s47. Historyczne wpływy mierzą pierwszorzędowe perturbacje wag AdamW przy wspólnych buforach; nie są idealną przyczynową dekompozycją rzeczywistego kroku, który zmienia także BatchNorm.

Regresja bezpośrednia estymuje zależność oczekiwanej zmiany na H od dostępnej historii. Nie wymaga wstawienia przyszłych u. Jest to aproksymacja warunkowej średniej, nie identyfikacja reakcji na dowolne zadane przyszłe sterowanie. Dla przyszłego sterowania trzeba jawnie oszacować rozkład przyszłych u z polityki i historii oraz propagować go przez model iteracyjny. Nie wolno podstawiać rzeczywistych przyszłych paczek, gradientów, wag ani późniejszych reszt.

## Reakcja na sterowanie i identyfikowalność

Dla estymowanej liniowej realizacji `q_(k+1)=Tq_k+Gu_k`, `y_k=Cq_k`, reakcja impulsowa to `C T^j G`. Rangi macierzy obserwowalności i sterowalności badają konkretną realizację. Dostępne paczki są ograniczone do realnego zbioru, dlatego pełna ranga algebraiczna nie oznacza możliwości fizycznego zadawania każdego wektora u.

Pilot nie estymuje minimalnej pełnej realizacji (θ,m,v,BN) i nie stwierdza jej obserwowalności ani sterowalności. Raportuje rangę oraz kondycję regresorów. Kolumny opisu paczki są współzależne, a lr stały w analizowanej fazie: ridge zapewnia obliczalny predyktor, lecz nie czyni wszystkich współczynników identyfikowalnymi. Współczynnik regresji historycznych paczek nie jest samodzielnym dowodem ich wpływu przyczynowego. Potrzebne byłyby bogatsze wzbudzenie i kontrolowane warianty polityki, przy zachowaniu wspólnych punktów pracy.

## Szum i niepewność

Reszta d_H-dhat_H zawiera przyszłe nieznane paczki, niedomknięcie stanu, nieliniowość i zmianę statystyk/parametrów. Proces Gaussian może mieć niezerową autokowariancję; białość i kształt rozkładu to oddzielne własności. Reszty z nakładających się celów H=500 są z definicji zależne. Dla krótszego H=10 rozstawienie startów co 10 usuwa nakładanie celów, ale nie gwarantuje stacjonarności ani niezależności stanu.

Raportujemy autokorelacje, skośność, kurtozę, korelacje z przeszłym s i pchnięciem oraz różnicę wariancji między fazami. Przedziały powstają z kwantyli błędów historycznej walidacji; ich pokrycie testowe jest miarą empiryczną, bez gwarancji. Nie narzucamy równowagowej relacji fluktuacja-dysypacja dla nieustacjonarnego AdamW.

## Wariant wybrany w tym pilotażu

Na H=500 spośród nowych rodzin walidacja wybrała samo oryginalne jądro z ridge 0.0001, bez dodatkowego u. Jest praktycznie równoważne działającemu GLE z ridge 0.001. Ogólny wybór obejmujący stare rodziny wybrał starego Koopmana. Nie wyprowadzono ani nie wykazano lepszego uniwersalnego równania H=500. [Wyniki wszystkich ramion](WYNIKI.md).
