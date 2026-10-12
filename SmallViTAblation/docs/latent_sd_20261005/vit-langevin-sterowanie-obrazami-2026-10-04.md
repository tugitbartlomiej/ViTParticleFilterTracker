# Sterowanie uczeniem ViT przez dobór obrazów z użyciem równania Langevina

Data: 2026-10-04. Status: plan badawczy, bez uruchomionych nowych obliczeń.

## Cel i pytanie

Czy model ruchu latentu pozwala wybierać paczki obrazów, których rzeczywiste użycie do uczenia ViT poprawia rozpoznawanie narzędzia bardziej niż dobór losowy i proste metody selekcji, przy porównywalnym koszcie?

Docelowa pętla użytkownika: przewidzieć korzystny ruch reprezentacji narzędzia, wyszukać lub wygenerować obrazy realizujące ten ruch, użyć ich do treningu i zmierzyć rzeczywisty efekt. Najpierw sprawdzamy wybór istniejących obrazów. Generowanie jest osobnym etapem, aby oddzielić jakość równania, wyszukiwania i generatora.

## Co wiadomo, a co pozostaje hipotezą

Dotychczasowe raporty znajdują się w osobnym workspace:
`F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/`.

| Dowód, ścieżka względem tego workspace | Znaczenie |
|---|---|
| `experiments/maly_vit_horyzont_rownania_krok/WYNIKI.md` | Wariant A: 94% redukcji błędu przyrostu siły tożsamości na 100 kroków, 93% na 150 i 82% na 500, z siłami obliczanymi przy rzeczywistych wagach kolejnych kroków. To rekonstrukcja z dostarczanym wymuszeniem, nie autonomiczna prognoza. Jedno ziarno, 180 sond, cztery początki. |
| `experiments/maly_vit_rownanie_latentu/WYNIKI_EDEN.md` | Dziesięć ziaren: około 94,9% mediany umiejętności jednego kroku z siłą paczki i pędu. Nie przenosić tej liczby na horyzont 100. |
| `experiments/maly_vit_langevin_kierunki_tozsamosci/WYNIKI.md` | Autonomiczny GLE dla 31 cząsteczek jednego obrazu w 8 kierunkach identyfikacji: H100 redukcja MSE 13,05%, znak współrzędnych 65,72%. Diagnostyka na 101 początkach, bez przedziału niepewności. |
| `experiments/maly_vit_petla_sd_rownanie/WYNIKI.md` | Pętla SD i wyszukiwania poprawiała szybkość uczenia wobec losowania, ale nie wykazano dodatkowej korzyści prognozy wobec SD bez prognozy. Brakowało kontroli wyszukiwania bez SD. |
| `docs/23_METODA_PRZEWIDYWANIA_LATENTU.md` | Definicja siły paczki, pędu i dekompozycji AdamW; pomiar wymaga dodatkowych przejść przez sieć. |

Liczby są odczytem istniejących raportów, nie nową replikacją. Dokładną implementację wariantu A, współczynniki i metrykę trzeba zweryfikować przed implementacją nowej metody. Nie utożsamiać prognozy średniej reprezentacji narzędzia z prognozą wszystkich jego cząsteczek.

Hipoteza H1: równanie porządkuje kandydackie paczki według rzeczywistego skutku jednej aktualizacji na obrazach kontrolnych.

Hipoteza H2: użycie tego rankingu daje poprawę jakości detekcji w zamkniętej pętli względem uczciwych kontroli.

Hipoteza H3: model przewidujący siły pozwala wybierać sekwencje paczek na 100 kroków bez dostępu do rzeczywistych przyszłych wag. H3 nie wynika z H1 ani z wyniku 94%.

## Stan, sterowanie i równanie

Cząsteczka `z_i(t)` jest reprezentacją tego samego tokenu tego samego stałego obrazu kontrolnego. W omawianym ViT ma 768 współrzędnych. Jeden obraz ma 600 tokenów; maska narzędzia wybiera ich podzbiór. Globalny układ jest wspólną bazą, nie średnią tokenów.

Do pomiaru tożsamości zamrażamy na danych uczących projekcję `P` lub zbiór natywnych kanałów `J`. Stan zadaniowy to `q_i=P^T z_i` lub `q_i=z_i[J]`. Raportujemy również błąd pełnego `z_i`, gdy model go przewiduje. Sama projekcja nie dowodzi wyłącznej semantycznej tożsamości narzędzia.

Sterowanie `B_t` obejmuje identyfikatory obrazów, etykiety, transformacje i ich ziarna. Stan sieci obejmuje wagi `theta_t`, optymalizator `o_t` i bufory `r_t`, w tym BatchNorm. Zdjęcia nie są bezpośrednio wektorem siły.

Schemat wariantu z mierzoną siłą paczki i pędem:

```text
g_t(B) = gradient_theta L_train(theta_t, r_t; B) po ustalonym clippingu
delta_theta_B, delta_theta_ped, delta_theta_decay = rozkład kroku AdamW
S_B,i = q_i(theta_t + delta_theta_B; r_t) - q_i(theta_t; r_t)
S_ped,i = q_i(theta_t + delta_theta_ped; r_t) - q_i(theta_t; r_t)

q_i(t+1) - q_i(t) = c_p S_ped,i + c_B S_B,i + epsilon_i
```

To schemat do odtworzenia z kodu, nie deklaracja, że w istniejącym wariancie wszystkie współczynniki były swobodnie dopasowywane. Suma części kroku wag może być dokładna, ale suma zmian nieliniowego latentu zwykle nią nie jest. Wpływ weight decay, zmian buforów i nieliniowych interakcji należy zmierzyć, zapisać jawnie lub przypisać do reszty.

Pęd AdamW jest zależny od mianownika aktualizacji. Jeżeli mianownik zawiera gradient kandydata, również `S_ped` zależy od `B`. Nie traktować go wtedy jako identycznego dla wszystkich paczek. Zamrożenie mianownika jest osobną aproksymacją i wymaga osobnego porównania.

Alternatywny empiryczny GLE z pamięcią:

```text
Delta q_i(t+1) = f(q_i(t))
               + sum_l K_l Delta q_i(t-l)
               + C u_i(theta_t, o_t, r_t; B_t) + epsilon_i(t+1)
```

`f` jest dopasowanym składnikiem dryfu, suma opisuje pamięć, a `u` wymuszenie obrazami. Wariant z historią przyrostów i wariant z jawnym pędem porównujemy osobno, aby nie policzyć tego samego wpływu dwukrotnie. Reszty nie nazywamy białym szumem ani nie zakładamy relacji fluktuacja-dyssypacja bez testów.

## Cel wyboru obrazów

Zamrożony probe `p_phi(q)` daje wynik obecności narzędzia na stałych sondach pozytywnych i negatywnych. Kalibrację oceniamy na wydzielonych danych. Bez kalibracji mówimy o wyniku klasyfikatora, nie o potwierdzonym prawdopodobieństwie.

```text
J(q) = mean_positive log p_phi(q)
     + lambda_neg mean_negative log(1 - p_phi(q))

B* = argmax_B [J(q_hat(t+1 | B)) - J(q(t))]
```

Obie klasy i filmy otrzymują ustalone wagi. Agregacja tokenów jest jawna; nie uznajemy tokenów jednego filmu za niezależne obserwacje. Wagi celu i ograniczenia powtórzeń ustalamy na walidacji. Najpierw maksymalizujemy zysk na jeden krok. Koszt obliczeń raportujemy osobno i wykonujemy dodatkowe porównanie przy równym budżecie czasu GPU.

Probe jest celem zastępczym. Ostateczny sukces wymaga poprawy rzeczywistej detekcji, nie tylko podnoszenia aktywacji wybranego kanału. Zbiór sond używany do selekcji jest częścią danych optymalizacji; nie wolno nazywać go końcowym testem.

## E1: ranking paczek i rzeczywisty skutek jednego kroku

1. Ustalić pulę 1500 obrazów, jej pochodzenie, split po filmach, etykiety i sampler. Sprawdzić w kodzie wcześniejszego treningu, czy paczki były losowane z/bez zwracania i kiedy następowało tasowanie; nie zakładać tego z rozmowy.
2. Zapisać kompletny stan ViT, optymalizatora, schedulerów, RNG i buforów. Dla pilota wybrać z góry 3 początki: wczesny, środkowy i późny trening.
3. Z puli treningowej utworzyć 32 kandydackie paczki po 32 obrazy na początek, z zamrożonymi transformacjami. Wszystkie selektory widzą identyczne kandydatury.
4. Policzyć siły przy aktualnych wagach i ranking równania. Zamrozić prognozy przed pomiarem skutków.
5. Na niezależnych kopiach tego samego stanu wykonać dokładnie jedną prawdziwą aktualizację dla każdego kandydata. Oceniać ten sam zestaw sond; zawsze odtwarzać pełny stan przed kolejnym kandydatem.
6. Zmierzyć korelację rang, zysk wybranej paczki względem losowej, regret względem najlepszego rzeczywistego kandydata, MSE ruchu, znaki i cosinus przyrostów. Rozdzielić metryki cząsteczek i agregacji.

Kontrole: losowy kandydat; największa strata treningowa; podobieństwo do sond bez równania; sam wpływ paczki bez pamięci; sam pęd; ranking z siłami przypisanymi do niewłaściwych kandydatów. Placebo zachowuje pulę i rozkład sił, niszczy tylko powiązanie paczka-skutek. Ujemny efekt placebo nie unieważnia automatycznie kontroli.

Jednokrokowe rzeczywiste rozgałęzienie jest kosztownym odniesieniem jakości i kosztu. Jeżeli selektor Langevina wymaga prawie tyle samo pracy, jego sens wymaga przewagi kosztowej albo innej mierzalnej korzyści. Nie prezentować wyroczni wykorzystującej przyszły wynik jako selektora o równym dostępie do informacji.

Obalenie H1: ranking nie przewyższa placebo/losowania na nowych początkach i ziarnach lub zysk celu zastępczego nie przekłada się na poprawę na niezależnych sondach.

## E2: rzeczywista pętla sterowania

Uruchomić dopiero po diagnostyce E1. Pilot 3 ziarna, potwierdzenie 10 nowych ziaren. Ramiona: losowe, największa strata, podobieństwo bez równania, siła paczki bez pamięci, pełny selektor. Jednakowy start wag, pula, liczba aktualizacji, wielkość paczki i ograniczenia użycia zdjęć. Każde ramię liczy siły przy własnych aktualnych wagach.

Pilot: 100 aktualizacji ViT, selekcja przed każdą aktualizacją. Pełny budżet i częstotliwość ewaluacji zamrozić przed potwierdzeniem na podstawie kosztu pilota. Raportować dodatkowo porównanie przy równym całkowitym czasie GPU, obejmującym gradienty kandydatów i pomiary sond.

Główna metryka: znormalizowane pole pod krzywą AP50 na wydzielonych filmach ewaluacyjnych. Dodatkowe: końcowe AP50, AP50:95, fałszywe wykrycia, kalibracja, zysk probe, różnorodność i powtórzenia danych oraz koszt. Test końcowy nie wpływa na wybór ramek, hiperparametry ani zatrzymanie.

Kryterium H2: dolna granica 95% przedziału sparowanej różnicy głównej metryki ponad zero wobec losowego i największej straty, z korektą Holma dla dwóch głównych porównań. Raportować efekty i rozrzut po ziarnach, nie tylko istotność. Brak przewagi nad samą siłą paczki oznacza brak wykazanego wkładu pamięci. Brak przewagi przy równym czasie GPU oznacza brak wykazanej korzyści obliczeniowej.

## E3: prognoza i planowanie stu kroków

Model przyszłego wymuszenia ma korzystać z aktualnego stanu, historii oraz cech zaplanowanych paczek. Uczyć go na wcześniejszych, kompletnie zarejestrowanych przebiegach i oceniać na nowych ziarnach/filmach. Znajomość przyszłych zdjęć jest dozwolona w prognozie warunkowej; nieznana przyszła kolejność wymaga prognozy po rozkładzie harmonogramów.

Porównać: historia bez sterowania; historia i znane cechy przyszłych paczek; rzeczywiste uczenie kopii sieci jako kosztowne odniesienie; siły z rzeczywistych przyszłych wag jako oddzielna diagnostyczna wyrocznia. Wyrocznia nie jest kandydatem do wdrożenia.

Horyzonty 1, 10, 50, 100 aktualizacji. W otwartym przebiegu po początku nie odświeżać stanów rzeczywistych. Osobno ocenić sterowanie z ponownym planowaniem po każdej aktualizacji; jego poprawności nie nazywać autonomiczną prognozą stu kroków.

Obalenie H3: brak poprawy MSE względem braku zmiany i historii albo brak poprawy rankingu sekwencji. Dobry wynik rekonstrukcji z wyrocznią nie zmienia tego werdyktu. Dopiero pozytywny H100 uzasadnia testy 150 i 500.

## E4: generowanie obrazów na podstawie celu latentu

Cel `q*` musi być osiągalny dla obrazu i sensowny semantycznie. Generator tworzy kandydatów, ViT weryfikuje ich reprezentacje, a selektor ocenia przewidywany skutek UCZENIA na nich. Bliskość latentu wygenerowanego obrazu do celu nie gwarantuje właściwego gradientu aktualizacji.

Kontrole: wyszukanie prawdziwych obrazów bez generatora; generator bez prognozy; generator z celem równania; losowy dobór. Wszystkie ramiona mają równy budżet danych oraz osobno równy budżet obliczeń. Sprawdzić etykiety syntetyków, geometrię narzędzia i artefakty. Nie nadawać automatycznie poprawnej etykiety tylko dlatego, że generator dostał taki warunek.

## Ustalona kolejność: selekcja archiwum, następnie generator

Doprecyzowanie użytkownika z 2026-10-04: pierwszym poziomem jest bezpośredni wybór istniejących zdjęć według korzystnego wpływu uczenia na latent tożsamości. Drugim poziomem mogą być wygenerowane warianty wybranych zdjęć. Nie potrzeba sukcesu autonomicznej prognozy H100, aby sprawdzić tę pętlę jednokrokową; E3 jest osobną gałęzią badań, nie warunkiem uruchomienia E4 po udanych E1/E2.

Przepływ: pula treningowa → kandydackie paczki → siły przy aktualnym stanie sieci → ranking przewidywanego zysku J → wybrana paczka → prawdziwa aktualizacja → pomiar efektu i odświeżenie rankingu. Nie maksymalizujemy samej normy ruchu: duży ruch może pogorszyć rozpoznawanie lub zwiększyć fałszywe wykrycia. Sondy obejmują narzędzie i tło, a testowe filmy pozostają poza selekcją.

W pilocie oceniamy paczki jako całość. Ranking pojedynczych zdjęć może później służyć jako etap wstępny, ale złożonej paczce trzeba ponownie policzyć siłę: AdamW, clipping i aktualizacja latentów są nieliniowe, więc suma osobnych ocen nie musi przewidywać skutku wspólnej paczki. Każdy kandydat jest oceniany od tego samego stanu; próbne oceny nie aktualizują produkcyjnego modelu.

Na drugim poziomie generator dostaje wybrane obrazy jako kotwice i tworzy warianty. Warianty przechodzą kontrolę etykiet, następnie tę samą ocenę wpływu aktualizacji co obrazy prawdziwe. Porównujemy bezpośrednio: wybrane oryginały, zwykłe augmentacje tych oryginałów i warianty generatora. Dodanie generatora wymaga wykazanej korzyści ponad dobór oryginałów i augmentacje, przy raportowanym pełnym koszcie.

Pominięcie generatora usuwa koszt syntezy, ale nie gwarantuje szybszego całego procesu. Mierzymy koszt oceny kandydatów, liczbę forward/backward i zysk jakości na sekundę GPU. Zmniejszenie puli kandydatów lub rzadsze odświeżanie rankingu dobieramy na pilocie, przed potwierdzeniem.

## Reprodukowalność i warunki startu

Przed wykonaniem założyć osobne rodziny badawcze E1-E4 przez narzędzia skillu `eksperyment`, z katalogami w `experiments/`, konfiguracją w `configs/` i wynikami w `outputs/`. Ten dokument jest planem, nie deklaracją istniejących wyników.

Zachować identyfikatory batchy, kolejność, transformacje, fingerprint splitów, probe i projekcję, stany optymalizatora, gradient clipping, bufory BN, siły, wszystkie rankingi kandydatów, rzeczywiste skutki, wersje kodu, środowisko i koszt. Dane z archiwum i wcześniejsze eksperymenty pozostają źródłami; nie przenosić ich w ramach tego planu.

Kontrole przed pełnym biegiem: odtwarzanie jednej aktualizacji z identycznego stanu; brak mutacji bazowego modelu podczas oceny kandydatów; zgodność jednostek sił; test, że podmiana przyszłych stanów nie zmienia prognozy; poprawne dzielenie po filmach; zgodność implementowanej metryki z definicją. Pilota użyć do pomiaru czasu i pamięci, potem ustalić zasoby Edena.

Najbliższy krok: audyt kodu dekompozycji AdamW i mały E1. Nie rozpoczynać od generowania ani prognozy 500 kroków. Najpierw rozstrzygnąć, czy równanie poprawnie wybiera jedną aktualizację o korzystnym rzeczywistym skutku.
