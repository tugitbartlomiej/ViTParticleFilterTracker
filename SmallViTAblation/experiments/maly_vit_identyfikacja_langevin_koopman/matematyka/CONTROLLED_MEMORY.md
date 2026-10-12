# Controlled drift i dwie pamięci: algebra oraz granica dostępności wejść

Sprawdzono rozdzielenie pamięci stanu i historycznych wejść w zadanym modelu liniowym, wspólną regresję blokową oraz zależność pamięci tylko od przeszłości. To kontrole algebraiczne modelu. Nie dowodzą liniowości latentu ViT, domknięcia controlled Koopmana, jakości predykcji ani identyfikowalności współczynników z konkretnego nagrania.

## Dokładna eliminacja macierzowa

Dla zgodnych wymiarów macierzy rozpatrujemy

$$h_{k+1}=E h_k+C x_k+F u_k,\qquad x_{k+1}=A x_k+B h_k+D u_k.$$

Wejście u_k działa na przejście od x_k do x_{k+1}. Iteracja pierwszego równania i podstawienie do drugiego dają

$$x_{k+1}=A x_k+D u_k+B E^k h_0+\sum_{j=0}^{k-1}B E^{k-1-j}C x_j+\sum_{j=0}^{k-1}B E^{k-1-j}F u_j.$$

W notacji wieku j≥0 definiujemy dwa oddzielne jądra:

$$K_j^x=B E^j C,\qquad K_j^u=B E^j F.$$

Zatem

$$x_{k+1}=A x_k+D u_k+B E^k h_0+\sum_{j=0}^{k-1}K_j^x x_{k-1-j}+\sum_{j=0}^{k-1}K_j^u u_{k-1-j}.$$

Indeks j=0 oznacza najnowszy historyczny stan x_{k−1} i najnowsze historyczne wejście u_{k−1}; ich współczynniki wynoszą BC i BF. Macierze mnożymy w zapisanej kolejności. Nie zakładamy odwracalności E, przemienności ani stabilności.

Dla k=0 obie sumy są puste. Dla k=2 otrzymujemy

$$x_3=A x_2+D u_2+B E^2h_0+BE C x_0+BC x_1+BE F u_0+BF u_1.$$

Jądra K^x i K^u mogą mieć różne rangi i rozmiary. Współdzielą propagator E w tym ukrytym modelu, ale swobodny fit opóźnień nie narzuca tej faktoryzacji automatycznie. Skończony model ukryty nie gwarantuje skończonej długości pamięci; zerowanie dalszych opóźnień wymaga dodatkowej struktury, np. nilpotentnego E, lub stanowi przybliżenie.

## Co wolno użyć w prognozie w chwili t

Przyjmujemy kontrakt informacyjny: znane są obserwacje x_0,…,x_t i wejścia u_0,…,u_{t−1}. Wejścia u_t,u_{t+1},… są nieznane. To indeksowanie oznacza, że ostatnie znane u_{t−1} należało do już zakończonego przejścia x_{t−1}→x_t.

Obie sumy pamięci w równaniu dla x_{t+1} używają dostępnej historii. Składnik D u_t pozostaje nieznany. Wyeliminowanie h nie usuwa go. W modelu matematycznym zmiana samego u_t o delta, bez zmiany przeszłości, pozostawia pamięć w t bez zmian, lecz zmienia x_{t+1} o D delta. Ten fakt sprawdzono symbolicznie. Stan h_0 lub jego efekt także wymaga estymacji, założenia początkowego albo fazy rozgrzewki.

Przy prognozie warunkowej odpowiedni nieznany wkład ma postać D E[u_t | historia do t], przy dodatkowym określeniu rozkładu wejść. Pominięcie tego wkładu nie jest algebraicznie równoważne modelowi źródłowemu. Jednokrokowa regresja wyłącznie na dostępnych opóźnieniach może estymować predykcję warunkową, ale jej współczynniki nie muszą odpowiadać fizycznym A,D,K^x,K^u. Dla horyzontu wielokrokowego trzeba określić i ocenić politykę prognozy przyszłych wejść; niedopuszczalne jest podawanie rzeczywistych przyszłych wejść z testowej trajektorii jako dostępnych cech.

Można zdefiniować model operacyjny

$$\widehat x_{t+1}=\widehat A x_t+\widehat D_- u_{t-1}+\sum_{l=1}^{L_x}\widehat K_l^x x_{t-l}+\sum_{l=2}^{L_u}\widehat K_l^u u_{t-l}.$$

Tutaj blok D_- opisuje najnowsze dostępne wejście; nie jest oryginalnym D działającym na u_t. Rozpoczęcie drugiej sumy od l=2 zapobiega duplikowaniu tej samej kolumny u_{t−1} w dwóch blokach. Równie poprawne jest połączenie wszystkich historycznych wejść w jeden blok od l=1 i całkowite usunięcie bloku D_-. Przycięcie historii, estymacja efektu h_0 i nieznane przyszłe u pozostają ograniczeniami tego modelu operacyjnego.

## Wspólna identyfikacja zamiast niezależnych fitów

Przy konwencji próbek w wierszach zbuduj jedną macierz Z przez konkatenację bloków drift, dostępne wejście, pamięć stanu oraz starsza pamięć wejść. Y zawiera target x_{t+1}. Wszystkie bloki współczynników składają się na W; odpowiedni operator działający na kolumnach ma transponowaną orientację.

Dla dodatnich kar blokowych i diagonalnej Lambda minimalizujemy

$$J(W)=\|Y-ZW\|_F^2+\operatorname{tr}(W^T\Lambda W).$$

Jednoczesne rozwiązanie spełnia

$$(Z^T Z+\Lambda)W=Z^T Y,\qquad W=(Z^T Z+\Lambda)^{-1}Z^T Y.$$

Dodatniość wszystkich kar zapewnia dodatnią określoność ZᵀZ+Lambda i jednoznaczność tego regularizowanego rozwiązania. Nie gwarantuje identyfikowalności modelu fizycznego. W implementacji numerycznej stosuje się rozwiązanie układu, zamiast jawnego tworzenia odwrotności.

Niech S=Lambda^{1/2}, Q=Z S^{−1}. Wtedy dokładna tożsamość skalowania kolumn wynosi

$$W=S^{-1}(Q^TQ+I)^{-1}Q^TY.$$

Wspólna macierz normalna zachowuje wyrazy mieszane między blokami. Niezależne dopasowanie każdego bloku do całego Y zwykle daje inne rozwiązanie. Wspólne dopasowanie zapobiega traktowaniu całego targetu jako osobnego celu każdego bloku, ale nie czyni jego rozkładu na bloki fizycznie jednoznacznym. Dobór opóźnień, kar i skalowania wymaga walidacji na danych treningowych, z zachowaniem chronologii.

## Weryfikacja SymPy

`controlled_verify.py` przeprowadza lokalne kontrole dokładnej arytmetyki:

- Pełne symboliczne macierze 2×2, niezależne wpisy wszystkich A,B,C,E,D,F, wejścia i warunki początkowe: równanie eliminacji z osobnymi sumami dla k=0,1,2,3.
- Zmiana u_t nie wpływa na sumę historycznych wejść w t; bezpośredni następny stan zmienia się o D delta.
- Ogólna symboliczna symetryczna macierz Grama dla dwóch skalarnych bloków z różnymi karami: dokładnie zerowa reszta równania normalnego, pod warunkiem niezerowego wyznacznika.
- Dokładna syntetyczna macierz Z o rozmiarze 10×6 i randze 6, target Y 10×2: bloki x_t, u_{t−1}, x_{t−1}, u_{t−2}. Użyte są kary 1,4,9,16; w sześciu kolumnach Lambda=diag(1,1,4,9,9,16). Największy użyty indeks wejścia to t−1.
- Reszta równania normalnego i gradient funkcji celu są zerowe; rozwiązanie przez skalowanie kolumn jest identyczne. Niezależne fity bloków dają inne rozwiązanie.

Syntetyczne targety nie są testem jakości prognozy ani trajektorią ViT. Tożsamość block ridge działa dla dowolnego takiego designu i nie jest dowodem trafności jego hipotezy. Nie przeprowadzano diagnostyki reszt, white noise, FDT ani filtracji Kalmana.

## Formalny zakres Lean

`ControlledMemory.lean` jest samodzielnym modułem opartym wyłącznie na `Init` i sprawdzonym w istniejącym lokalnym Lean 4.28.0. Przez indukcję dowodzi siedmiu twierdzeń nad Int:

1. `memory_add`: pamięć sumy wymuszeń jest sumą osobnych pamięci dla dowolnego k.
2. `memory_prefix`: pamięć w k zależy tylko od wartości q_j dla j<k.
3. `hidden_representation`: ukryta rekurencja jest sumą propagowanego warunku początkowego i pamięci dla dowolnego k.
4. `propagated_power`: propagowany warunek początkowy wynosi e^k h_0.
5. `controlled_elimination`: obserwowana rekurencja rozdziela pamięć c*x od pamięci f*u dla dowolnego k.
6. `input_memory_causal`: dowolna zmiana wejść od k wzwyż nie zmienia pamięci wejść w k, przy tej samej wcześniejszej historii.
7. `kernel_three`: dla trzech historycznych wymuszeń jawne współczynniki pamięci wynoszą b e²g, b e g i b g. Podstawienie g=c lub g=f daje oba skalarne jądra.

Formalizacja obejmuje całkowite współczynniki i skalary, rekurencyjną pamięć dla wszystkich k i rozwiniętą sumę dla k=3. Nie obejmuje ogólnego twierdzenia macierzowego nad R ani problemu optymalizacji ridge. Te części mają wyprowadzenie oraz oddzielne kontrole SymPy. W końcowym pliku nie ma `sorry`, `admit` ani deklarowanych `axiom`. Audyt pokazuje standardowe `propext` dla twierdzeń korzystających z upraszczania; dwa twierdzenia o prefiksie nie zależą od żadnych aksjomatów. Kompilacja: exit 0, bez ostrzeżeń.

## Reprodukcja

```powershell
& 'F:/Python/3.13/python.exe' 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_identyfikacja_langevin_koopman/matematyka/controlled_verify.py'
```

Rezultat `controlled_verification_result.json` zawiera kontrole, wersje, pełne komendy Lean, kody wyjścia, stdout, stderr oraz SHA-256 obu źródeł. `all_checks_passed=true` odnosi się do wymienionych kontroli algebraicznych i kompilacji. Zachowano wszystkie wcześniejsze artefakty matematyczne. Nie instalowano narzędzi, nie wywoływano cloud, nie zmieniano plików poza nowymi artefaktami tego katalogu.
