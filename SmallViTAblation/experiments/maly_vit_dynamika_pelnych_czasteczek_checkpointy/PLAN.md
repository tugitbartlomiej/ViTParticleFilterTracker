# Prognoza pełnych cząsteczek na stałych obrazach

Data: 2026-10-01. Rodzina: `maly_vit_dynamika_pelnych_czasteczek_checkpointy`. Klasa: badawcza, pilot eksploracyjny. Autoryzacja użytkownika: lokalne forwardy i małe dopasowanie, bez uczenia ViT, nowych zadań Eden i commitów.

## Pytanie i zakres pilota

Czy historia pełnego wektora cząsteczki pozwala przewidzieć jej następny stan lepiej niż zachowanie ostatniej pozycji albo ostatniej prędkości? Czy wspólna liniowa redukcja przestrzeni 768D, dopasowana wyłącznie do przeszłości, poprawia prognozę całego wektora?

Cząsteczka jest pojedynczym tokenem tego samego, stałego obrazu. Zachowujemy jego indeks i wszystkie 768 współrzędnych. Makrowektory to wspólne osie `e_j`, mikrowektory to składowe `z_ij e_j`. Nie utożsamiamy PCA z tożsamością narzędzia. Nie mierzymy tutaj zmiany wejścia między klatkami ani skutku usunięcia narzędzia przez LaMa.

## Dane i oszczędność zasobów

- Istniejący bieg Eden `2026-10-01_maly_vit_ruch_czastek_gesta_historia`, seed 47, AdamW, batch 32, kroki 0..1000. Jest to jeden przebieg optymalizatora.
- 101 sprawdzonych checkpointów co 10 kroków zawiera wagi i bufory BatchNorm. Plik `LATENT_GESTY.npy` zawiera już 1001 stanów pełnych reprezentacji: 12 stałych obrazów, 600 tokenów, 768 współrzędnych, float32.
- Pilot używa **101 stanów co 10** dla porównywalności z checkpointami. Pobraną gęstą historię czytamy przez mmap. Nie tworzymy drugiej pełnej kopii ani cache 2,2 GB.
- Oryginalne maski wyznaczają 218 tokenów z pokryciem większym od zera. Zachowujemy wszystkie osobno. Wybrana tablica 101×218×768 zajmuje około 67,6 MB RAM; nie zapisujemy jej na dysku.
- Obrazy i maski z pomiaru lokalnego są identyczne bajtowo ze zdalnymi wejściami; sprawdzono SHA256. Metadane przebiegu bierzemy z biegu gęstego, nie z wcześniejszego eksperymentu dwóch checkpointów.
- Budżet dodatkowego dysku po pobraniu: poniżej 20 MB. Twardo zachować minimum 1 GiB wolnego miejsca. CPU maksymalnie 4 wątki, planowany RAM poniżej 2 GB. GPU tylko do małego testu zgodności, po audycie; LaMa pracuje na CPU.

## Etap 0: wiarygodność pomiaru

1. Sprawdzić zdalny i lokalny SHA256 całego pliku, format NumPy, shape, dtype i metadane.
2. Porównać lokalny forward checkpointów 0 i 1000 z odpowiednimi stanami gęstej historii na 12 identycznych obrazach. Użyć `eval`, `no_grad`, float32, bez autocast i TF32, bez augmentacji. Odczyt: `bloki[-1].mlp.act` po GELU. Kolejność tokenów: `y*30+x`.
3. Powtórzyć forward i sprawdzić niezmienność buforów BN. Zapis wyniku ma podać błąd maksymalny i względny, wersje bibliotek, hash kopii modelu. Różnice sprzętu/PyTorch mogą wykluczyć zgodność bitową.
4. Kryterium pilota: powtórzenia zgodne numerycznie, brak NaN/Inf, brak zmiany BN, względny błąd L2 względem historii ≤1e-4. Jeżeli nie, zatrzymać interpretowanie prognoz do wyjaśnienia. Pełną historię można nadal bezpiecznie pobrać i zweryfikować.

## Etap 1: pełne wektory i redukcja liniowa

Najpierw porównanie autonomiczne. Nie ma potwierdzonego, zapisanego dziennika wykonanych minibatchy tego konkretnego biegu. Sam seed, przepis i straty co 50 kroków nie zastępują takiego dziennika. Nie wolno podłączyć historii sterowań z innego biegu s47/s48. Brak sterowania nie blokuje prognoz opartych na historii stanu.

Każdy model widzi tylko stany do początku prognozy włącznie. Maska oraz stały obraz są dostępne od początku. Model nie dostaje przyszłego gradientu, paczki, stanu AdamW, obrazu ani przyszłego pomiaru tokenów.

**Modele i podłogi:**

1. Persystencja: przyszła pozycja równa ostatniej pozycji. Zachowuje całą reprezentację i strukturę przestrzenną.
2. Stała prędkość: kontynuacja ostatniej różnicy między checkpointami. Dodatkowa kontrola: prędkość wygaszana współczynnikiem 0,8 na krok obserwacji, bez dopasowania na przyszłości.
3. Pełny ridge AR(1): osobny regularizowany liniowy predyktor przyrostu 768D każdej cząsteczki, uczony na tym samym oknie jej przeszłości. Implementacja dualna nie identyfikuje dowolnej macierzy 768×768 z 40 próbek; jest rozwiązaniem silnie ograniczonym danymi i karą.
4. Wspólna baza PCA 768×r dopasowana do odchyleń poszczególnych cząsteczek od ich własnych średnich w oknie przeszłości; `r` z {4, 8, 16}. Zachowujemy odrębny stan każdej cząsteczki. W tej bazie dopasowujemy osobny ridge AR(1), a opcjonalnie AR(2) z jednym opóźnieniem przyrostu. Prognoza jest rekonstruowana do pełnych 768D; składowa ortogonalna do PCA pozostaje na wartości początkowej prognozy.

To pilot predyktorów poszczególnych cząsteczek ze wspólną redukcją. Nie jest to jeszcze identyfikacja pełnej macierzy oddziaływań między cząsteczkami. Uogólnienie do wspólnego operatora z pamięcią wymaga osobnego porównania po wyniku pilota.

## Podział czasu i brak przecieku

- Okno obserwacji: ostatnie 41 stanów. Wszystkie dopasowywane modele używają tych samych **39 przejść** od lokalnego stanu 1 do 39 ku stanom 2 do 40. Lokalny stan 0 dostarcza poprzedniego przyrostu modelowi AR(2).
- Walidacja ustawień: początki prognoz 400 i 500, horyzonty 10, 20, 50 i 100. Wszystkie jej cele są dostępne najpóźniej w kroku 600.
- Test: początki 600, 700, 800 i 900; te same horyzonty. Parametry wyboru modelu zamrażamy przed testem. Operatory i PCA można ponownie dopasować wyłącznie do przeszłości danego początku, bez wykorzystania jego przyszłych celów.
- Ridge: jawna siatka siły kary {0,001; 0,1; 10}, przeskalowana energią macierzy wejściowej. Ranga PCA i kara wybierane wyłącznie na walidacji; AR(1)/AR(2) raportowane osobno.
- Krok 1000 jest końcem dostępnych danych. Nie ma prawdziwych celów po nim i nie wolno przedstawiać tego pilota jako potwierdzonej prognozy dalszego uczenia po rozgrzewce.

Podział `fit/holdout` zapisany przy obrazach dotyczył wcześniejszego pytania o semantyczne kierunki ruchu. Ten pilot prognozuje historię tych samych 12 obrazów. Raportuje również wyniki osobno dla tych grup, lecz **nie nazywa dopasowanych na ich własnej przeszłości predyktorów generalizacją na całkowicie nieznane obrazy**.

### Doprecyzowanie po przeglądzie matematycznym Astra

Pełny model ma postać `Δz_(k+1) = B(z_k−μ_X)+b`. Jest to regresja przyrostu na pozycję, nie autoregresja samych przyrostów. Ograniczenie rangi dotyczy korekty `B`; operator stanu `I+B` może mieć pełną rangę. Intercept jest średnim przyrostem, a wejścia i cele są centrowane tą samą regułą w pełnej i zredukowanej przestrzeni.

Macierz jądra każdego bloku wejściowego dzielimy przez jej średnią energię `trace(K)/N`, a następnie dodajemy `alpha I`. Wariant z pamięcią uśrednia tak znormalizowane jądra pozycji i ostatniego przyrostu. Dzięki temu siła kary jest względna do obserwowanej energii, nie do nominalnej liczby 768 lub r współrzędnych. Wszystkie warianty mają tę samą siatkę alpha.

Wspólna PCA wykorzystuje odchylenia od osobnych średnich cząsteczek oraz wagi: każdy obraz 1/12, tokeny w obrazie proporcjonalnie do pokrycia maską. Średnie, baza i operatory są zamrożone w obrębie każdego rolloutu. Nie ma teacher forcing. Rekonstrukcja jest dokładnie `zhat_(t+h)=z_t+U(xhat_(t+h)−x_t)`. Horyzonty 10/20/50/100 aktualizacji odpowiadają 1/2/5/10 krokom modelu.

Do oceny dodajemy normę prawdziwego przemieszczenia poza PCA: `||(I−UUᵀ)(z_(t+h)−z_t)||`. Przy ortonormalnym U kwadrat całkowitego błędu rozkłada się na błąd wewnątrz PCA i tę składową. Jest to podłoga reprezentacyjna obliczana wyłącznie do oceny, bez dostępu predyktora do przyszłości.

## Ocena i obalenie

Główny błąd to **RMS norm błędów pełnych wektorów**: najpierw średnia kwadratów norm w obrazie, następnie średnia po obrazach i na końcu pierwiastek. W głównej wersji tokeny w obrazie są ważone udziałem pokrycia oryginalną maską; raport obejmie również równe wagi tokenów. Dodatkowo podamy zwykłą średnią norm L2, aby pokazać wpływ dużych błędów. Wyniki cząsteczek pozostają dostępne osobno. To doprecyzowanie, wskazane w audycie kodu, zapisano przed odczytem testu i wykonaniem prognoz.

Raportować błąd względny do persystencji, błąd przewidzianego przemieszczenia, cosinus przemieszczenia tam, gdzie normy nie są bliskie zeru, oraz stabilność przy różnych horyzontach. Rozdzielić sam błąd reprezentacji od błędu jej niewielkiego ruchu, aby duża statyczna norma nie maskowała złej prognozy.

218 tokenów i ich okna nie są 218 niezależnymi treningami sieci. Jeden seed ogranicza wniosek do tego przebiegu. Nie stosować testów istotności opartych na traktowaniu każdego tokenu/czasu jako niezależnej próby. Podstawowy raport ma być opisowy, z wynikami po obrazach i początkach prognozy.

Hipotezę osłabia brak przewagi na odłożonych horyzontach nad persystencją i prędkością, niestabilny rollout, albo przewaga widoczna tylko po dopasowaniu PCA na całej przyszłej trajektorii. Tego ostatniego wariantu nie dopuszczamy do uczciwego rankingu.

## Ograniczenia organizacyjne

W tym repozytorium nie ma modułu rejestru `src.narzedzia.sprawdz_eksperymenty` wskazanego w skillu projektu grantowego. Deklaracja rodziny pozostaje lokalnie w `WYNIKI.json`; nie modyfikujemy rejestru innego projektu. Bez nowego treningu ViT, pobierania stanu optymalizatora, zadań Eden, commitów i zmian w badaniu 400 par LaMa.
